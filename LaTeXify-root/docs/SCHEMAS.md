# LaTeXify JSON Schemas

The project now validates planner artifacts, consensus bundles, and layout blueprints using Pydantic models.  The generated JSON Schema documents below capture the contract for each structure and can be referenced by upstream tools or additional validators.

## PlanTaskSchema

```json
{
  "title": "PlanTaskSchema",
  "description": "Schema describing a single plan task.",
  "type": "object",
  "properties": {
    "id": {
      "title": "Id",
      "minLength": 1,
      "type": "string"
    },
    "title": {
      "title": "Title",
      "default": "",
      "type": "string"
    },
    "kind": {
      "title": "Kind",
      "type": "string"
    },
    "content_type": {
      "title": "Content Type",
      "type": "string"
    },
    "order": {
      "title": "Order",
      "minimum": 0,
      "type": "integer"
    },
    "layout_block_id": {
      "title": "Layout Block Id",
      "type": "string"
    },
    "block_id": {
      "title": "Block Id",
      "type": "string"
    },
    "asset_path": {
      "title": "Asset Path",
      "type": "string"
    },
    "notes": {
      "title": "Notes",
      "type": "object"
    }
  },
  "required": [
    "id"
  ]
}
```

## ConsensusBlockSchema

```json
{
  "title": "ConsensusBlockSchema",
  "description": "Schema describing post-judge consensus blocks.",
  "type": "object",
  "properties": {
    "block_id": {
      "title": "Block Id",
      "type": "string"
    },
    "text": {
      "title": "Text",
      "default": "",
      "type": "string"
    },
    "block_type": {
      "title": "Block Type",
      "default": "text",
      "type": "string"
    },
    "page_index": {
      "title": "Page Index",
      "default": 0,
      "minimum": 0,
      "type": "integer"
    },
    "flagged": {
      "title": "Flagged",
      "default": false,
      "type": "boolean"
    },
    "ocr_outputs": {
      "title": "Ocr Outputs",
      "type": "object",
      "additionalProperties": {
        "type": "string"
      }
    }
  },
  "required": [
    "block_id"
  ]
}
```

## LayoutBlueprintSchema

```json
{
  "title": "LayoutBlueprintSchema",
  "description": "Schema for layout planner blueprints.",
  "type": "object",
  "properties": {
    "version": {
      "title": "Version",
      "type": "string"
    },
    "model_name": {
      "title": "Model Name",
      "type": "string"
    },
    "created_at": {
      "title": "Created At",
      "type": "string"
    },
    "plan": {
      "title": "Plan",
      "type": "object"
    },
    "raw_response": {
      "title": "Raw Response",
      "type": "string"
    },
    "source": {
      "title": "Source",
      "type": "object"
    },
    "warnings": {
      "title": "Warnings",
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": [
    "version",
    "model_name",
    "created_at",
    "plan"
  ]
}
```

## PlanSchema

```json
{
  "title": "PlanSchema",
  "description": "Schema describing the full planner output consumed downstream.",
  "type": "object",
  "properties": {
    "doc_class": {
      "title": "Doc Class",
      "type": "string"
    },
    "doc_class_hint": {
      "title": "Doc Class Hint",
      "type": "object"
    },
    "frontmatter": {
      "title": "Frontmatter",
      "type": "object"
    },
    "content_flags": {
      "title": "Content Flags",
      "type": "object",
      "additionalProperties": {
        "type": "boolean"
      }
    },
    "tasks": {
      "title": "Tasks",
      "type": "array",
      "items": {
        "$ref": "#/definitions/PlanTaskSchema"
      }
    }
  },
  "definitions": {
    "PlanTaskSchema": {
      "title": "PlanTaskSchema",
      "description": "Schema describing a single plan task.",
      "type": "object",
      "properties": {
        "id": {
          "title": "Id",
          "minLength": 1,
          "type": "string"
        },
        "title": {
          "title": "Title",
          "default": "",
          "type": "string"
        },
        "kind": {
          "title": "Kind",
          "type": "string"
        },
        "content_type": {
          "title": "Content Type",
          "type": "string"
        },
        "order": {
          "title": "Order",
          "minimum": 0,
          "type": "integer"
        },
        "layout_block_id": {
          "title": "Layout Block Id",
          "type": "string"
        },
        "block_id": {
          "title": "Block Id",
          "type": "string"
        },
        "asset_path": {
          "title": "Asset Path",
          "type": "string"
        },
        "notes": {
          "title": "Notes",
          "type": "object"
        }
      },
      "required": [
        "id"
      ]
    }
  }
}
```
