# Final G-code failure-segment pipeline

This folder is the final operational lineage, copied from `reprint_or_overlay_bundle.zip`.
The detector is the simple YCrCb pixel-based extractor, not the pseudo-Voigt/Yen branch in `../tests/`.

## Execution order

1. **`filament_array.py`** — rotate/crop each frame and create a pixel mask for likely no-filament/background using the YCrCb score.
2. **`gcode_coordinate_mask.py`** — parse extrusion geometry from G-code and rasterize the selected layer in printer/world coordinates.
3. **`merge_miss_print_or.py`** — restore every frame mask to camera geometry, place it using the filename pose, OR observations globally, and intersect no-filament evidence with expected G-code filament.
4. **`extractor_segment_reprint_or_notebook.ipynb`** — score actual G-code segments from the missed-print mask and draw segments selected for reprinting.
5. **`extractor_segment_reprint_or_copy_images_notebook.ipynb`** — additionally find and copy the original frames that evaluated each faulty area.
6. **`extractor_segment_reprint_or_overlay_copy_images_notebook.ipynb`** — additionally project the selected global faulty segments back onto the original images with the inverse merge geometry.

The three notebooks are successive versions. The overlay/copy notebook is the most complete output workflow; the earlier notebooks remain useful for simpler runs and debugging.

## Supporting examples

- `example_mask_triptych.png` — expected filament, OR no-filament evidence, and their intersection.
- `example_reprint_segments_raw.png` — raw segment-level reprint result.
- `example_reprint_segments_labeled.png` — labeled segment result.
- `example_merged_no_filament_or.png` — global OR merge.
- `example_miss_print_or.png` — G-code-expected pixels classified as no filament.
- `example_expected_filament.png` — rasterized expected print path.

## Important configuration

The notebooks contain machine-specific paths and constants that must be changed for a new run. The most important are the image folder, G-code path, output/cache paths, layer index, coordinate flips, camera scale, line width, and severity threshold. The bundle used approximately `26.13 px/mm` and a `0.45 mm` expected line width.

The input image filenames are expected to follow:

```text
frame_<frame>_t_<time>_x_<x>_y_<y>_z_<z>.<extension>
```

Install dependencies with:

```bash
pip install -r requirements.txt
```
