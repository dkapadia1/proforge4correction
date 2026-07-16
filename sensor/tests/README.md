# Experimental and testing code

This folder contains detector experiments, abandoned alternatives, calibration scripts, and earlier merge/scoring versions. It is organized by the corresponding section of `../readme/README.md` rather than by original ZIP location.

The source files are preserved without behavioral edits. Many contain machine-specific absolute paths, disabled branches, notebook-state assumptions, or imports written for their original flat directories. They document what was tried; they are not all expected to run unchanged.

## Directory map

| Directory | General README section |
|---|---|
| `centerline_and_column_models/` | **Laser analysis → Centerline extraction** and **Column based models** |
| `template_matching/` | **Ineffective models → Template matching/full distribution comparison** |
| `pseudo_voigt_digital_twin/` | **Ineffective models → Psuedo-voigt digital twin** and **Future paths → Digital twin with inferred centers** |
| `gcode_merge_and_segment_experiments/` | **Using the result → GCODE extraction / Determine what is wrong / Smooth by segment** |
| `calibration_annotation_and_debug/` | Supporting work for **Klipper internals**, **Better data**, geometry calibration, and visual validation |
| `fixtures/` | Sample G-code used by geometry and digital-twin experiments |
| `legacy_notes/` | Original brief notes from the earlier `server` folder |

## Centerline extraction and column-based models

| File | README section | What it tested |
|---|---|---|
| `centerline_and_column_models/analyze_five.py` | **Column based models → Initial gaussian sigma analysis** | Fits a Gaussian to each column, smooths the sigma sequence, and detects likely filament region boundaries from width changes. Includes thresholded and global segment variants. |
| `centerline_and_column_models/analyze_sevon.py` | **Centerline extraction → An assumption of saturation** and **Column based models** | Replaces a pure Gaussian with a robust pseudo-Voigt fit so broad shoulders and mixed Gaussian/Lorentzian profiles can be represented. This is the early per-column fitting version. |
| `centerline_and_column_models/analyze_six.py` | **Centerline extraction**, **Initial gaussian sigma analysis**, and **Change in skew** | Main combined research file: adaptive MRSAC filtering, Hessian/Zhao ridge extraction, binary fallback centerlines, smoothing, distribution fitting, skew confidence, and split/underextrusion analysis. |
| `centerline_and_column_models/analyze_eight.ipynb` | **Centerline extraction** and **Column based models** | Notebook experiments around MRSAC kernel behavior, column widths, Gaussian residual maps, and pixel-level visualization of skew contributions. |
| `centerline_and_column_models/numeric_segment.py` | **Column based models → Initial gaussian sigma analysis** | Avoids fitting every column by baseline subtraction and numeric FWHM. Combines width, amplitude, SNR, local stability, neighboring agreement, and optional robust fit quality. |
| `centerline_and_column_models/contrast_segmentation.py` | **Centerline extraction → An assumption of saturation** | Short-lived saturation experiment: least-saturated channel selection, unsaturated-wing fitting, gradient-edge width, second-moment width, and saturation penalties. It was not integrated; the final confidence accidentally emphasized amplitude instead of the calculated width. |
| `centerline_and_column_models/combine_methods.py` | **Column based models → Change in skew** and **Future paths → Combining priors** | Combines sigma-derived filament regions, skew/cubic split confidence, brightness, amplitude, and intended two-peak evidence. The detailed two-peak implementation is bypassed by an early return. |
| `centerline_and_column_models/accuracy_local.py` | **Change in skew** | Runs split-confidence methods across pose-encoded frames and interpolates a spatial heatmap, allowing the skew/split score to be inspected over the print. |
| `centerline_and_column_models/server.py` | Supporting utility for **Centerline extraction** | Small HTTP service that runs the centerline analysis on a local image and returns maximum centerline deviation. |
| `centerline_and_column_models/__init_.py` | Supporting file | Empty package marker retained from the original server folder. |

## Template and full-distribution matching

| File | README section | What it tested |
|---|---|---|
| `template_matching/analyze_nine.ipynb` | **Ineffective models → Template matching/full distribution comparison** | Canonical median templates, raw and derivative matching, high-pass filtering, affine amplitude/baseline alignment, residual maps, and multiple profile-distance comparisons. |
| `template_matching/analyze_ten.ipynb` | Same section | Incomplete notebook containing only imports; retained because it marks a started follow-up experiment but has no recoverable method. |
| `template_matching/filament_array.py` | Same section | Early attempted reusable extractor using empty/full derivative templates and `full_match * (1 - empty_match)`. The saved file is syntactically incomplete at `photos[ ]`, showing it was an abandoned edit rather than a final implementation. |

## Pseudo-Voigt digital twin and Yen-difference branch

| File | README section | What it tested or upgraded |
|---|---|---|
| `pseudo_voigt_digital_twin/pseudo_voigt_laser_mixer.ipynb` | **Psuedo-voigt digital twin** | Initial synthetic laser renderer using fitted empty/full pseudo-Voigt profiles and a local expected-filament mask. Originally fit many columns. |
| `pseudo_voigt_digital_twin/pseudo_voigt_laser_mixer_aligned_empty.ipynb` | **Psuedo-voigt digital twin** | Upgrades generation to a median profile and explicitly recenters the mismatched empty reference. |
| `pseudo_voigt_digital_twin/pseudo_voigt_laser_mixer_gcode_mask.ipynb` | **Psuedo-voigt digital twin** and **Using the result → GCODE extraction** | Creates the expected local filament mask from G-code and uses it to render an ideal laser image. |
| `pseudo_voigt_digital_twin/mask_to_laser.py` | **Psuedo-voigt digital twin** | Extracted reusable implementation of pseudo-Voigt fitting, template alignment, laser ROI geometry, mask rotation, and synthetic laser rendering. |
| `pseudo_voigt_digital_twin/offset_tuning.ipynb` | **Psuedo-voigt digital twin** | Scans the empty-template center offset, compares rendered output, and converges near the later `-26 px` correction. |
| `pseudo_voigt_digital_twin/filament_array_offset_yen.py` | **Psuedo-voigt digital twin** | Compares the real frame with the G-code-driven synthetic image, applies Yen thresholding and connected-component cleanup, and returns a no-filament mask. |
| `pseudo_voigt_digital_twin/filament_yen_fast.py` | Same section | Cached/optimized version of the synthetic rendering and Yen mask extraction. |
| `pseudo_voigt_digital_twin/merge_yen_fast.py` | **Using the result → Determine what is wrong** | Fast global merging specialized for the cached Yen detector, including cropped-mask pasting. |
| `pseudo_voigt_digital_twin/merge(6).py` | **Psuedo-voigt digital twin** and **Determine what is wrong** | Transitional global merge that ORs detector masks and can also build per-frame expected G-code masks using nearest-Z layer selection. |
| `pseudo_voigt_digital_twin/gcode_expected_print_mask.py` | **Using the result → GCODE extraction** | Early local expected-print mask generator selecting the G-code layer nearest the frame Z. |
| `pseudo_voigt_digital_twin/gcode_expected_print_mask(3).py` | Same section | Exact duplicate of `gcode_expected_print_mask.py`, retained to preserve the downloaded filename expected by fallback import code. |

## G-code, global merge, and segment-scoring experiments

| File | README section | What it tested or upgraded |
|---|---|---|
| `gcode_merge_and_segment_experiments/gcode_expected_print_mask_layer_index.ipynb` | **Using the result → GCODE extraction** | Changes expected-layer selection from nearest recorded Z to explicit layer index and provides visual tuning of pose, orientation, scale, and line width. |
| `gcode_merge_and_segment_experiments/early_vote_merge/merge.py` | **Determine what is wrong** | Earliest global assembly: accumulates no-filament votes and observation counts, then thresholds their ratio. This predates the physical OR assumption. |
| `gcode_merge_and_segment_experiments/simple_detector_merge/merge.py` | **Determine what is wrong** | Later merge using the simple detector and restored camera masks; supports both vote and OR canvases but does not yet contain the final G-code-coordinate result object. |
| `gcode_merge_and_segment_experiments/merge_miss_print_coord.py` | **GCODE extraction** and **Determine what is wrong** | Joins the simple detector’s global coordinate merge with `gcode_coordinate_mask.py`, retaining coordinate metadata and producing expected, covered, no-filament, and missed-print masks. It is the direct precursor to the final OR bundle merger. |
| `gcode_merge_and_segment_experiments/extractor_segment_reprint_threshold_notebook.ipynb` | **Smooth by segment** | Transitional evidence-ratio notebook. Scores segments using mean, upper percentile, fractions over evidence thresholds, neighborhood smoothing, and severity quantiles before the final OR notebook simplified the score. |

The final `gcode_coordinate_mask.py`, `merge_miss_print_or.py`, and OR notebooks are in `../model/`, not duplicated here.

## Calibration, annotation, and debugging

| File | README section | What it did |
|---|---|---|
| `calibration_annotation_and_debug/pick_pixel_matches.py` | Supporting **Using the result → GCODE extraction** | Interactive correspondence picker between consecutive frames. Compares pixel displacement with filename X/Y movement to estimate one camera scale. |
| `calibration_annotation_and_debug/pixel_matches.csv` | Same section | Saved click correspondences and coordinate deltas used to estimate approximately `26.13 px/mm`. |
| `calibration_annotation_and_debug/centerline_radius_annotator.py` | **Future paths → Better data** and **Combining priors** | Manual tool for marking centerline points and local radii, interpolating a tube mask, and comparing it with the simple YCrCb detector. Useful for generating spatial ground truth. |
| `calibration_annotation_and_debug/make_debug_video.py` | **Determine what is wrong** | Produces synchronized videos of the original frame, restored detector mask, and growing global merged result, including the current camera location. |
| `calibration_annotation_and_debug/make_laser_image.py` | Supporting **Template matching** / **Psuedo-voigt digital twin** | Small early geometry check that rotates a rectangular laser ROI into image coordinates and compares masked frames. |

## Fixture and notes

| File | Purpose |
|---|---|
| `fixtures/P4_one_layer_annular_disc_60OD_12ID_0p20H.gcode` | One-layer annular-disc G-code used by expected-mask, coordinate, and digital-twin tests. |
| `legacy_notes/server_README.txt` | Original brief descriptions of the Gaussian, centerline, skew, and HTTP-server files. |
| `requirements.txt` | Consolidated packages imported by the experimental scripts and notebooks. |

## Important archival notes

- Generated PNGs, caches, bytecode, and the two very large debug MP4 files were intentionally omitted. They are outputs, not testing code, and can be regenerated from the scripts when source image data is available.
- Absolute Windows paths and hardcoded frame indices were preserved. Replace them before running old experiments.
- The experimental files are organized for comprehension. A few imports assumed the old flat directory layout, so running a historical branch may require adding its sibling folders to `PYTHONPATH` or editing the import path.
