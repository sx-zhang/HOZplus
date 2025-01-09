# Semantic Map Processing

## Overview

This repository provides tools for processing semantic maps by generating individual graphs for each scene and merging them into a comprehensive graph. The workflow involves downloading a semantic map archive, extracting its contents, generating k-means based graphs for each dataset, and finally merging all scene graphs into a single unified graph.

## Dataset

You can download the semantic maps dataset for Gibson and AI2THOR from [semantic_maps](https://!!!!!!!!!!).

The folder should look like this
```python
  generate_hoz_graphs/ 
    ├── gen_hoz_graph_ai2thor
    ├── gen_hoz_graph_gibson
    ├── semantic_maps/
        ├── ai2thor
            ├── bathrooms
                ├── FloorPlan401_1_20_graph.pbz2
                ├── FloorPlan401_5_20_graph.pbz2
                ├── ...
            ├── bedrooms
            ├── kitchens
            ├── living_rooms
        ├── gibson
            ├── Allensville.h5
            ├── Beechwood.h5
            ├── ...
```

## Generating Graphs

### Gibson
1. `python gen_k_means_gibson.py`
2. `python merge_all.py`

You should find the merged graph at `generate_hoz_graphs/gen_hoz_graph_gibson/saved_graphs/merged_gibson_hoz_graph.pbz2`.

### AI2THOR
1. `python gen_k_means_ai2thor.py`
2. `python merge_all.py`

You should find the merged graph at `generate_hoz_graphs/gen_hoz_graph_ai2thor/saved_graphs/merged_ai2thor_hoz_graph.pbz2`.
