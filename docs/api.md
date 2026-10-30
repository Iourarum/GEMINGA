# API reference

## Budget

```{eval-rst}
.. autoclass:: geminga.Budget
   :members:
```

## Opening

```{eval-rst}
.. autofunction:: geminga.open
.. autofunction:: geminga.view
.. autofunction:: geminga.open_gef
.. autofunction:: geminga.open_gem
.. autofunction:: geminga.open_stereoseq
.. autofunction:: geminga.open_h5ad
.. autofunction:: geminga.open_fastx
.. autofunction:: geminga.open_arrow
.. autofunction:: geminga.open_image
.. autofunction:: geminga.open_hub
.. autofunction:: geminga.open_urls
.. autofunction:: geminga.open_paths
.. autofunction:: geminga.index_fastx
.. autofunction:: geminga.index_gem
```

## Views

```{eval-rst}
.. autoclass:: geminga.View
   :members:
.. autoclass:: geminga.RowView
   :members:
.. autoclass:: geminga.SeqView
   :members:
.. autoclass:: geminga.ImageView
   :members:
.. autoclass:: geminga.Overview
   :members:
.. autoclass:: geminga.Extent
   :members:
.. autoexception:: geminga.CostError
```

## Stereo-seq

```{eval-rst}
.. autoclass:: geminga.GefFile
   :members:
.. autoclass:: geminga.StereoSeqView
   :members:
.. autoclass:: geminga.GemView
   :members:
.. autoclass:: geminga.StereoSeqDataset
   :members:
```

## Format readers and sinks

```{eval-rst}
.. autoclass:: geminga.H5adReader
   :members:
.. autoclass:: geminga.ArrowIPCReader
   :members:
.. autoclass:: geminga.TiledImageReader
   :members:
.. autoclass:: geminga.TiffTileArray
   :members:
.. autoclass:: geminga.ParquetSink
   :members:
.. autoclass:: geminga.MaskSink
   :members:
.. autoclass:: geminga.ProgressiveCanvas
   :members:
```

## Orchestration

```{eval-rst}
.. autofunction:: geminga.orchestrator
.. autoclass:: geminga.Orchestrator
   :members:
.. autoclass:: geminga.Piece
   :members:
.. autoclass:: geminga.Resolution
   :members:
.. autoclass:: geminga.Blueprint
   :members:
.. autoclass:: geminga.Puzzle
   :members:
.. autoclass:: geminga.Coverage
   :members:
.. autofunction:: geminga.read_descriptions
.. autofunction:: geminga.find_accessions
.. autofunction:: geminga.resolve_pieces
```

## Harmonisation

```{eval-rst}
.. autoclass:: geminga.Harmonizer
   :members:
.. autoclass:: geminga.Vocabulary
   :members:
.. autoclass:: geminga.GeneMap
   :members:
.. autoclass:: geminga.Alignment
   :members:
.. autoclass:: geminga.ColumnMap
   :members:
.. autofunction:: geminga.id_type
```

## Preprocessing out of core

```{eval-rst}
.. autofunction:: geminga.prep
.. autoclass:: geminga.Prep
   :members:
.. autoclass:: geminga.PrepResult
   :members:
```

## The journal of streamed pieces

```{eval-rst}
.. autofunction:: geminga.journal
.. autofunction:: geminga.use_journal
.. autoclass:: geminga.Journal
   :members:
.. autoclass:: geminga.JournalEntry
   :members:
.. autoclass:: geminga.DatasetTrail
   :members:
.. autofunction:: geminga.journal_recording
```

`geminga._core.TrigramIndex` (the Rust core; `Journal.index()` returns one): `build(path, block_lines=64)`, `load(path)`, `save(path)`, `update()`, `search(pattern, ignore_case=True)`, `candidates(pattern)`, and the read-only `source_path`, `indexed_bytes`, `n_lines`, `n_blocks`, `block_lines`, `n_trigrams`, `n_everywhere`, `postings_bytes`.

## Compressed blocks and the shared GPU

```{eval-rst}
.. autofunction:: geminga.compressed_store
.. autoclass:: geminga.CompressedStore
   :members:
.. autoclass:: geminga.BitpackCodec
   :members:
.. autoclass:: geminga.BytesCodec
   :members:
.. autoclass:: geminga.Encoded
   :members:
.. autoclass:: geminga.DeviceArray
   :members:
.. autofunction:: geminga.check_codec
.. autofunction:: geminga.register_codec
.. autofunction:: geminga.get_codec
.. autofunction:: geminga.encode_csr
.. autofunction:: geminga.decode_csr
.. autofunction:: geminga.reserve
.. autoclass:: geminga.Reservation
   :members:
.. autoclass:: geminga.ReservationServer
   :members:
.. autoclass:: geminga.Policy
   :members:
```

## Rendering

```{eval-rst}
.. automodule:: geminga.render
   :members: navigator, chip, hits, progressive
```

## Helpers

```{eval-rst}
.. autofunction:: geminga.print_cost
.. autofunction:: geminga.print_plan
.. autofunction:: geminga.print_summary
.. autofunction:: geminga.sample_table
.. autofunction:: geminga.iter_numpy
.. autofunction:: geminga.iter_torch
```
