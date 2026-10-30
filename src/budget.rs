//! The governor. Every byte that becomes resident is accounted against a tier before it is
//! materialized, and released when the consumer is done with it.
//!
//! Three tiers, each an independent pool:
//!   - `ram`   host memory holding decoded chunks
//!   - `vram`  device memory (accounting only — the caller does the actual cudaMalloc via
//!     torch/cupy; GEMINGA just refuses to let it over-commit)
//!   - `spill` scratch space on disk for chunks you want to keep but not hold in RAM
//!
//! Backpressure, not OOM: a request larger than the whole tier fails immediately with a message
//! that says what to change. A request that merely doesn't fit *right now* blocks until another
//! consumer releases, or times out. That distinction is what makes an 8 GB laptop behave
//! predictably instead of dying halfway through an epoch.

use crate::error::{GemingaError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone, Copy)]
pub struct TierStats {
    pub capacity: u64,
    pub used: u64,
    pub peak: u64,
    pub grants: u64,
    pub waits: u64,
    pub wait_millis: u64,
    pub denials: u64,
}

#[derive(Debug, Default)]
struct TierState {
    used: u64,
    peak: u64,
    grants: u64,
    waits: u64,
    wait_millis: u64,
}

#[derive(Debug)]
pub struct Tier {
    pub name: String,
    pub capacity: u64,
    state: Mutex<TierState>,
    cv: Condvar,
    denials: AtomicU64,
}

impl Tier {
    pub fn new(name: &str, capacity: u64) -> Self {
        Self {
            name: name.to_string(),
            capacity,
            state: Mutex::new(TierState::default()),
            cv: Condvar::new(),
            denials: AtomicU64::new(0),
        }
    }

    /// Reserve `n` bytes. Blocks while the tier is full; fails fast if `n` can never fit.
    pub fn acquire(&self, n: u64, timeout: Duration) -> Result<()> {
        if n > self.capacity {
            self.denials.fetch_add(1, Ordering::Relaxed);
            return Err(GemingaError::Budget(format!(
                "cannot fit {} in the '{}' tier (capacity {}). Lower the chunk size \
                 (chunk_bytes / batch_size), or raise the budget for this tier.",
                human(n),
                self.name,
                human(self.capacity)
            )));
        }
        let mut st = self.state.lock().unwrap();
        if st.used + n <= self.capacity {
            st.used += n;
            st.peak = st.peak.max(st.used);
            st.grants += 1;
            return Ok(());
        }
        // Doesn't fit now: wait for a release. Every release wakes every waiter, so the
        // deadline is measured from the first wait, not restarted on each wake.
        let start = Instant::now();
        st.waits += 1;
        loop {
            let remaining = timeout.saturating_sub(start.elapsed());
            let (guard, res) = self.cv.wait_timeout(st, remaining).unwrap();
            st = guard;
            if st.used + n <= self.capacity {
                st.used += n;
                st.peak = st.peak.max(st.used);
                st.grants += 1;
                st.wait_millis += start.elapsed().as_millis() as u64;
                return Ok(());
            }
            if res.timed_out() || start.elapsed() >= timeout {
                st.wait_millis += start.elapsed().as_millis() as u64;
                self.denials.fetch_add(1, Ordering::Relaxed);
                return Err(GemingaError::Budget(format!(
                    "timed out after {:?} waiting for {} in the '{}' tier ({} of {} in use). \
                     Something is holding chunks: release them, shrink the chunk size, \
                     or raise the budget.",
                    timeout,
                    human(n),
                    self.name,
                    human(st.used),
                    human(self.capacity)
                )));
            }
        }
    }

    /// Reserve `n` bytes only if they are free right now: true when taken, false otherwise.
    /// One lock, so two callers cannot both see the same free bytes and both proceed.
    pub fn try_acquire(&self, n: u64) -> bool {
        if n > self.capacity {
            self.denials.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let mut st = self.state.lock().unwrap();
        if st.used + n <= self.capacity {
            st.used += n;
            st.peak = st.peak.max(st.used);
            st.grants += 1;
            true
        } else {
            false
        }
    }

    /// Give bytes back and wake one waiter.
    pub fn release(&self, n: u64) {
        let mut st = self.state.lock().unwrap();
        st.used = st.used.saturating_sub(n);
        drop(st);
        self.cv.notify_all();
    }

    pub fn stats(&self) -> TierStats {
        let st = self.state.lock().unwrap();
        TierStats {
            capacity: self.capacity,
            used: st.used,
            peak: st.peak,
            grants: st.grants,
            waits: st.waits,
            wait_millis: st.wait_millis,
            denials: self.denials.load(Ordering::Relaxed),
        }
    }

    pub fn available(&self) -> u64 {
        let st = self.state.lock().unwrap();
        self.capacity.saturating_sub(st.used)
    }

    pub fn reset_peak(&self) {
        let mut st = self.state.lock().unwrap();
        let used = st.used;
        st.peak = used;
    }
}

#[derive(Debug)]
pub struct BudgetInner {
    pub ram: Tier,
    pub vram: Tier,
    pub spill: Tier,
    /// How long to block before giving up.
    pub timeout: Duration,
}

impl BudgetInner {
    pub fn tier(&self, name: &str) -> Result<&Tier> {
        match name {
            "ram" => Ok(&self.ram),
            "vram" => Ok(&self.vram),
            "spill" => Ok(&self.spill),
            other => Err(GemingaError::Invalid(format!(
                "unknown tier {other:?}; expected 'ram', 'vram' or 'spill'"
            ))),
        }
    }

    pub fn acquire(&self, tier: &str, n: u64) -> Result<()> {
        self.tier(tier)?.acquire(n, self.timeout)
    }

    pub fn try_acquire(&self, tier: &str, n: u64) -> Result<bool> {
        Ok(self.tier(tier)?.try_acquire(n))
    }

    /// The chunk a reader should plan when the caller gave no size: a quarter of what is free,
    /// floored at 1 MB but never above what is free (the "chunk floor rule").
    pub fn default_chunk(&self) -> u64 {
        let avail = self.ram.available();
        (avail / 4).max((1u64 << 20).min(avail)).max(1)
    }

    pub fn release(&self, tier: &str, n: u64) {
        if let Ok(t) = self.tier(tier) {
            t.release(n);
        }
    }
}

/// Held by a reader for the lifetime of one resident chunk. Releasing is idempotent.
#[derive(Debug)]
pub struct Lease {
    budget: Option<Arc<BudgetInner>>,
    tier: String,
    bytes: u64,
    released: bool,
}

impl Lease {
    pub fn new(budget: Option<Arc<BudgetInner>>, tier: &str, bytes: u64) -> Self {
        Self {
            budget,
            tier: tier.to_string(),
            bytes,
            released: false,
        }
    }

    /// A lease that accounts nothing — used when no budget is attached.
    pub fn none() -> Self {
        Self {
            budget: None,
            tier: String::new(),
            bytes: 0,
            released: true,
        }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn release(&mut self) {
        if !self.released {
            if let Some(b) = &self.budget {
                b.release(&self.tier, self.bytes);
            }
            self.released = true;
        }
    }

    /// Resize a live lease to what was actually materialized: give back the surplus, or take
    /// the shortfall (which may wait, like any acquire).
    pub fn adjust(&mut self, actual: u64) -> Result<()> {
        if self.released {
            return Ok(());
        }
        if let Some(b) = &self.budget {
            if actual < self.bytes {
                b.release(&self.tier, self.bytes - actual);
            } else if actual > self.bytes {
                b.acquire(&self.tier, actual - self.bytes)?;
            }
        }
        self.bytes = actual;
        Ok(())
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.release();
    }
}

pub fn human(n: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut x = n as f64;
    for (i, u) in U.iter().enumerate() {
        if x < 1024.0 || i == U.len() - 1 {
            return if i == 0 {
                format!("{n} B")
            } else {
                format!("{x:.1} {u}")
            };
        }
        x /= 1024.0;
    }
    unreachable!()
}

/// Parse "8GB", "512MB", "1.5 GiB", or a plain byte count.
pub fn parse_size(s: &str) -> Result<u64> {
    let t = s.trim().to_ascii_lowercase().replace(' ', "");
    if let Ok(n) = t.parse::<u64>() {
        return Ok(n);
    }
    let (num, mult) = if let Some(p) = t.strip_suffix("tib").or_else(|| t.strip_suffix("tb")) {
        (p, 1u64 << 40)
    } else if let Some(p) = t.strip_suffix("gib").or_else(|| t.strip_suffix("gb")) {
        (p, 1u64 << 30)
    } else if let Some(p) = t.strip_suffix("mib").or_else(|| t.strip_suffix("mb")) {
        (p, 1u64 << 20)
    } else if let Some(p) = t.strip_suffix("kib").or_else(|| t.strip_suffix("kb")) {
        (p, 1u64 << 10)
    } else if let Some(p) = t.strip_suffix('b') {
        (p, 1u64)
    } else {
        (t.as_str(), 1u64)
    };
    let v: f64 = num
        .parse()
        .map_err(|_| GemingaError::Invalid(format!("cannot parse size {s:?}")))?;
    if v < 0.0 {
        return Err(GemingaError::Invalid(format!("negative size {s:?}")));
    }
    Ok((v * mult as f64) as u64)
}
