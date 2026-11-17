# MathVision 3D

**Draw on paper → show it to the webcam → detect → select → extract → explore in 3D.**

MathVision 3D is a desktop laboratory that turns hand-drawn geometric figures and
function graphs into interactive mathematical objects. A triangle sketched on paper
becomes a measurable triangle (or a prism) in a 3D workspace. A sketched sine wave
becomes `y = A·sin(Bx + C) + D` with fitted parameters, a derivative, tangent lines
and, if you choose, a generated surface.

The interface is in Georgian; code and documentation are in English.

---

## 1. Installation (Windows)

Requirements: Windows 10/11, Python 3.11–3.12 (3.12 recommended), a webcam
(optional: images and video files work too). No GPU, cloud service or paid API is needed.

```bat
setup.bat
```

The script creates `.venv`, installs `requirements-gestures.txt` (all core packages plus
MediaPipe) and writes the sample images to `samples/`. To set things up by hand:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tools\generate_samples.py
```

Optional hand-gesture control:

```bat
.venv\Scripts\python -m pip install -r requirements-gestures.txt
.venv\Scripts\python tools\download_hand_model.py
```

The second command downloads `hand_landmarker.task` (about 7.8 MB) from Google's
MediaPipe model storage into `models/`.

## 2. Running

```bat
run.bat                                   :: webcam (first camera found)
run.bat --image samples\sine_with_axes.jpg
run.bat --sample triangle                 :: built-in synthetic drawing
run.bat --video samples\demo_webcam.mp4   :: a video file stands in for the webcam
run.bat --camera 1                        :: choose a camera index
run.bat --smoke-test out.png              :: automated end-to-end check, then exit
```

### Workflow

1. **Camera** panel (left): start the camera, or load an image, sample or video.
2. Hold the paper in front of the camera. Detected objects are outlined live and
   labelled, for example `#2 სამკუთხედი 96%`. The paper sheet is outlined in cyan and
   detected coordinate axes in magenta.
3. **Select** an object by clicking it in the video or in the *აღმოჩენილი ობიექტები* list.
   The selection follows the object while the paper moves.
4. Press **გადაღება და ამოღება** (*Capture & Extract*). The frame is frozen and the
   selected object is placed in the 3D workspace.
5. Explore it in the **3D viewport** (centre):
   * **Camera mode**: left-drag orbits, middle or Shift-drag pans, the wheel zooms,
     and a click selects an object.
   * **Object mode**: left-drag rotates the selected object, Shift or middle-drag moves
     it, right-drag rolls it, and the wheel scales it.
   * Toolbar buttons: reset camera, front/top/side/isometric views, orthographic
     projection, grid, axes, auto-rotate.
6. The **right panel** shows the type, equation, confidence and fit error, candidate
   models and measurements. It also has the display mode, rotation/position/scale,
   rendering style, colours, line width, opacity and calibration controls.

If detection fails, change the threshold mode, sensitivity or contrast. You can also
use **რეგიონის ხელით არჩევა** (*manual region*) and drag a rectangle around the drawing.
The *Binary* preview shows exactly what the recogniser sees.

### Display modes

| Object | Mode A | Mode B | Mode C |
|---|---|---|---|
| Triangle / polygon / circle / ellipse | flat shape in 3D | **extrusion** (prism / cylinder, adjustable depth) | **geometric analysis** (vertex names, side lengths, angles, radii) |
| Function graph (sine, parabola…) | 2D curve in the XY plane (+ derivative, tangent, samples) | **space curve** `(x, f(x), z(x, y, t))` with a typed z-expression | **surface**: `z = f(x)` over a y-range, `f(x)·f(y)`, radial `f(r)`, or a surface of revolution |
| Line / arc / freehand curve | curve | ribbon extrusion | analysis |

Extrusions, space curves and surfaces are always labelled as **user-selected
mathematical transformations**. A single 2D sketch does not determine a unique 3D object.

### Calibration and honesty of measurements

* Without calibration, coordinates are **normalised**: the object's largest extent spans
  4 units. Every measurement is shown as *ნორმალიზებული — შეფასებითი* (normalised,
  estimated).
* **Calibrate** by giving the real length of a reference measure: side AB for polygons,
  the diameter for circles, the major axis for ellipses, the length for lines, the chord
  for arcs, or the horizontal span for curves. You can also give the x value where the
  curve starts. After that, measurements carry your unit and the equation is refitted.
* When drawn coordinate axes are detected, their crossing becomes the origin. The phase
  of a sinusoid is then meaningful, and the curve is reported as **sin** or **cos**.
  Without axes it is reported as a generic sinusoid, with a note that the phase is
  relative to an estimated origin.
* If no known family fits well, no equation is invented. The curve is kept as a
  smoothing spline and labelled as a freehand curve.

## 3. Architecture

```
mathvision3d/
  main.py                CLI entry point (python -m mathvision3d)
  i18n.py                Georgian UI strings
  synthetic.py           synthetic hand-drawn test scenes + demo video generator
  smoke_test.py          scripted end-to-end GUI check (--smoke-test)
  camera/
    camera_manager.py    CameraWorker (QThread: webcam or video file), CameraProbe
    frame_processor.py   ProcessingWorker: runs the pipeline on the latest frame only
  vision/                ── image -> DetectedObject (pixel space) ──
    preprocessing.py     shadow removal, contrast, denoise, adaptive/Otsu/manual threshold
    perspective_corrector.py  paper quad detection + homography rectification
    curve_extractor.py   skeleton centre-line ordering, coordinate-axis detection/removal
    shape_detector.py    closed/open stroke classification
    recognizer.py        Recognizer protocol + registry (plug-in point for future AI models)
    pipeline.py          VisionPipeline: frame -> DetectionResult (+ work->frame homography)
    models.py            ShapeKind, DetectedObject, AxesInfo, DetectionResult
  mathematics/           ── pure maths, no UI/CV dependencies ──
    geometry.py          areas, angles, circle/ellipse fits, vertex refinement
    function_fitting.py  linear/abs/quadratic/cubic/sine fitting + model selection
    transformations.py   Transform3D (rotation matrix, position, scale)
    coordinates.py       CoordinateSystem: pixels <-> local units, calibration state
    expressions.py       AST-validated evaluation of user formulas
  core/
    math_object.py       MathObject: detection + coordinates + fit + display/transform state
  rendering/
    object_renderer.py   MathObject -> PyVista meshes and labels for each display mode
    scene_manager.py     plotter scene: axes, grid, actors, selection, picking, export
    interaction.py       custom VTK interactor style (camera/object modes)
  interface/             PySide6 widgets: main window, camera panel, viewport, controls
  gestures/
    hand_tracker.py      MediaPipe HandLandmarker worker + GestureInterpreter state machine
tests/                   pytest suite (geometry, fitting, recognition, rendering, gestures)
tools/                   generate_samples.py, download_hand_model.py
samples/                 sample images + demo_webcam.mp4
```

Data flow:

```
CameraWorker ──frame──▶ ProcessingWorker ──DetectionResult──▶ MainWindow ──▶ CameraPanel (overlays)
                         (VisionPipeline)                         │
                                                       Capture & Extract
                                                                  ▼
                              MathObject(detection, axes) ──▶ SceneManager / ObjectRenderer ──▶ VTK
                                    ▲                                 ▲
                              ControlsPanel                 ViewportInteraction / gestures
```

Image recognition, mathematical reconstruction, rendering and UI are separate layers.
`vision` and `mathematics` do not import Qt or VTK, so they can be tested and reused
headlessly.

### Recognition pipeline

1. **Paper detection**: Otsu on the blurred frame, then the largest bright convex
   quadrilateral, then a homography. The image is warped to an upright sheet and a 2 %
   margin is cropped.
2. **Binarisation**: the image is divided by a morphological background estimate (this
   removes shadows and gradients). Then a contrast stretch, Gaussian blur, an adaptive
   threshold (or Otsu or manual), opening/closing and small-component removal.
3. **Axes**: Hough segments are merged per line, and a long horizontal and a long vertical
   line must cross *inside* both segments. Rectangle corners therefore do not count. The
   axes are erased, and curve pieces that crossed them are re-joined by in-band growth.
   A join is made only when the growth links two different ink components, so tick
   marks stay separate.
4. **Classification** of each connected component:
   * **Closed** (large hole, or a filled blob, or an open stroke whose ends almost meet):
     an ellipse model is compared with a polygon model (approxPolyDP, then merging of
     collinear and short edges, then **line-fit refined vertices**). Inner and outer
     outlines are averaged to get the stroke centre line. Four corners near 90° give a
     rectangle or square.
   * **Open**: the stroke is skeletonised, then ordered by the longest geodesic path
     (two BFS passes). Then it is tested in order: straight line (PCA residual), then
     circular arc (Kasa fit, sweep angle), then graph `y = f(x)` (x-monotonic, a family
     is recognised), otherwise a freehand curve.
5. **Function fitting**: samples are binned per x. Five families are fitted. Sine
   frequency candidates come from the FFT, zero crossings and fixed periods. They are
   pre-screened by linear least squares and refined with Levenberg–Marquardt using an
   analytic Jacobian. A sine must show ≥ 0.6 periods, which prevents degenerate "sine as
   parabola" fits. The simplest family within 15 % of the best NRMSE wins. **Confidence**
   combines fit quality with separation from competing (non-superset) families.

Typical processing time is 35–90 ms per frame at a 960 px working size on a laptop CPU.

### Extending with AI recognition

Implement `vision.recognizer.Recognizer.recognize(work_bgr, binary, stroke_width, params)`
so it returns `DetectedObject`s. Then call `register_recognizer("my_model", factory)` and
construct `VisionPipeline(recognizer=create_recognizer("my_model"))`. Everything after
detection stays unchanged.

## 4. Gesture control (optional, experimental)

Enable **ჟესტების ჩართვა** in the camera panel while the camera is running.

| Gesture | Effect |
|---|---|
| pinch (thumb + index) held ≈ 3 frames | grab the selected object |
| move the pinching hand | rotate (or translate, depending on the mode) |
| two pinching hands, change distance | scale |
| open hand / hand leaves view | release |

Unwanted motion is suppressed with hysteresis on the pinch ratio, a minimum hold time,
a handedness-confidence threshold, exponential smoothing and a dead zone.
Gestures never block the mouse workflow. If MediaPipe or the model file is missing, the
toggle explains why and turns itself off.

## 5. Testing

```bat
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m mathvision3d --smoke-test %TEMP%\mv3d.png
```

The **pytest** suite (48 tests) covers:

* geometry, transforms and the expression sandbox;
* recovery of sine parameters (clean and noisy), sin/cos naming, family selection
  (linear, |x|, x², x³), rejecting random curves, derivatives;
* recognition on synthetic drawings: triangle, rotated triangle, circle, ellipse,
  rectangle, sine, noisy sine with shadow, sine over drawn axes (with calibration giving
  `2·sin(x)`), parabola, four objects side by side, perspective-distorted sine and
  triangle, line and arc, manual region, and blank, dark, noise and empty images;
* rendering of every display mode, prism closure, picking, and export with transform;
* the gesture state machine.

The **smoke test** drives the real window. It streams `samples/demo_webcam.mp4` as a live
camera, tracks a selection across moving frames, then runs capture & extract, the
extrusion, analysis, derivative, tangent and surface modes, a viewport click, mouse-style
rotate/scale, screenshots, the space-curve expression and rejection of an unsafe
expression. Finally it checks that the panel layouts fit.

## 6. Known limitations

* **Overlapping** strokes merge into one connected component. Draw objects separately,
  or use the manual region tool.
* Axis detection expects a "+"-shaped frame (the axes cross inside both segments).
  L-shaped axes are treated as ordinary lines.
* Numerical tick labels are not read (no OCR). Calibrate through the right panel instead.
* Perspective correction needs the whole sheet to be visible against a darker
  background. Otherwise the frame is processed as is.
* The project pins `opencv-contrib-python<5`. OpenCV 4.14 is the version the full stack,
  MediaPipe included, was tested with.
* Live-webcam capture could not be exercised on the development machine, because its
  integrated camera was not present to Windows. The live path was verified with a video
  file streamed through the same `CameraWorker` (`--video`, and the smoke test).
