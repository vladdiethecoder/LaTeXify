## Plan & Graph Schema Notes

### plan.json

Each entry is a `common.PlanBlock` serialized to JSON:

- `block_id`: stable identifier used throughout the pipeline.
- `chunk_id`: pointer into `chunks.json`.
- `label`: human readable heading or caption.
- `block_type`: `section`, `paragraph`, `figure`, `table`, `equation`, etc.
- `images`: extracted image paths tied to the block.
- `metadata`: layout metadata copied from ingestion/layout. Existing keys include
  `region_type`, `header_level`, `list_depth`, `formula_detected`, `table_signature`.

New hierarchical information is stored in this metadata object:

- `hierarchy_level`: `"part" | "chapter" | "section" | "subsection"` for heading blocks.
- `hierarchy_parent`: `block_id` of the containing heading or `null` for the root.
- `hierarchy_path`: ordered list of titles from the document root to the current block.
- `resolved_label`: generated LaTeX label used by the cross-reference system (figures/tables/equations).

These keys are optional and treated as extension points, so existing consumers continue
to work even if they ignore them.

### graph.json

The structure graph emitted by `structure_graph.py` contains:

- `nodes`: serialized `GraphNode` objects with:
  - `node_id`, `type`, `label`
  - `metadata`: original plan metadata plus `hierarchy` (with `level` + `path`) when
    available, `chunk_id`, `page`, and `images`.
  - `children`: IDs of structural children.
  - `parent`: parent node ID (omitted for the root).
- `edges`: two edge types
  - `"hierarchy"` edges connect parent and child sections/blocks.
  - `"order"` edges connect consecutive blocks to preserve reading order.
- `reading_order`: simple list of plan block IDs.

Consumers that only depended on the original `"order"` edges can keep doing so, while
agents that need richer structure can opt-in to the new `parent` pointer and
`"hierarchy"` edges.
