# RL_training 19 — תיקון יציבות BBOX ופתיחת מסלול הנחיתה

## מה התברר מהצילום ומהצלבת הקוד

התיקון הקודם לא פתר את מסלול הנחיתה משום שהוא פתח את Gate ה־Catch-up, אך השאיר Gate שני שהיה עדיין מסוגל לחסום כל ירידה.

### 1. `bottom_bbox_rel_err` יצר Deadlock בגובה

המדד מחושב בקירוב כך:

```text
bbox_rel_x = 2 * distance_from_image_center_px / bbox_width_px
```

בצילום, מרכז הרכב נמצא קרוב יחסית למרכז התמונה (`center_error` בסדר גודל של 0.15), אבל רוחב ה־BBOX קטן ביחס לתמונה. לכן `bbox_rel_err` יכול להיות בערך 1.2–1.3, בזמן ש־Landing Lock דרש לכל היותר 0.55.

כלומר:

```text
צריך לרדת כדי שה־BBOX יגדל
אבל אסור לרדת כל עוד ה־BBOX היחסי קטן מדי
```

זו הסיבה לכך שנשאר `LOCK_PENDING(0/2)` או `ALIGN_BBOX`, גם כאשר `raw_vz_action` ביקש ירידה.

### 2. ה־BBOX הגולמי של YOLO שימש ישירות לבקרה

`_strict_track()` החזיר בכל פריים את ה־BBOX הגולמי שנבחר, ו־`_observe()` השתמש בו מיד עבור:

- שגיאת המרכז;
- נגזרת תנועת התמונה;
- Kalman וחיזוי;
- בקר ה־XY;
- Observation;
- הציור בחלון Agent 2.

ResNet יכול לאשר בזהות גבוהה גם BBOX מלא של הרכב וגם BBOX חלקי/רחב של אותו רכב. לכן זהות נכונה לא מבטיחה גאומטריית BBOX יציבה. קפיצה אחת הפכה מיד לקפיצת מהירות ולפקודת XY הפוכה.

### 3. Agent 1 צייר BBOX חדש על פריים תחתון ישן

ה־snapshot המשותף הכיל BBOX אך לא את הפריים המדויק שעליו חושב. Agent 1 צייר אותו על `_cached_downward_frame`, שעלול היה להילקח בזמן אחר. זה מסביר את הצילום שבו המלבן הירוק נמצא ימינה מהרכב.

### 4. חלון freshness של 1.25 שניות היה קצר ממחזור העיבוד

כאשר Agent 1 סיים את עיבוד ה־Front לאחר יותר מ־1.25 שניות, הוא יכול היה לדחות את ה־snapshot של Agent 2 ולהפעיל שוב Tracker תחתון עצמאי. כך חזרה כפילות התפיסה והתקבלו שני BBOX-ים מזמנים שונים.

### 5. חיבור ה־XY המקבילי לא כלל Slew Limit

Feed-forward, תיקון Bottom ו־Agent 1 חוברו ונחתכו רק לפי מהירות כוללת. קפיצת BBOX אחת הייתה יכולה לשנות את וקטור המהירות בכמה מטרים לשנייה בצעד יחיד.

## השינויים שבוצעו

### BBOX נפרד לזהות ולבקרה

ה־BBOX הגולמי נשאר במסלול ResNet/identity כדי לא לשנות את זיהוי המטרה. נוסף `control_bbox` נפרד עבור גאומטריה ובקרה:

- Median קצר של מדידות LIVE אחרונות;
- EMA נפרד למרכז ולגודל;
- Alpha נמוך יותר כאשר מזוהה קפיצת מרכז/גודל;
- בזמן PRED נשמר ה־BBOX המסונן האחרון, במקום לגזור מהירות מ־BBOX גולמי לא יציב.

### Gate גובה דו־שלבי

מעל `1.50m` מעל המטרה:

- Landing Lock משתמש ב־image-center error המסונן;
- `bbox_rel_err` אינו Gate, משום שהוא תלוי באופן מלאכותי בגודל ה־BBOX.

מתחת ל־`1.50m`:

- Gate ה־BBOX היחסי חוזר לפעול;
- כך נשמרת דרישה שמרכז המצלמה יהיה בתוך footprint הרכב לקראת מגע.

ב־Catch-up:

- רכישה: `center_error <= 0.24` במשך שני פריימי LIVE מאומתים;
- שמירת lock: `center_error <= 0.34`;
- predicted center מעל `0.34` עדיין חוסם ירידה;
- הירידה נשארת מוגבלת ל־`0.28m/s`, ומתחת למטר ל־`0.12m/s`.

### שיתוף perception עקבי

ה־snapshot של Agent 2 כולל כעת:

- BBOX מסונן;
- אותו פריים BGR שעליו הוא חושב;
- timestamp של התצפית עצמה.

Agent 1 מעתיק את אותו פריים ל־debug overlay. חלון ה־freshness הותאם ל־`4.0s`, מעל מחזור העיבוד שנמדד, כדי שלא תופעל בטעות רשת Bottom שנייה.

### ריסון פקודות XY

רק כאשר Bottom guidance פעיל:

- LIVE speed cap: `1.60m/s`;
- PRED speed cap: `1.10m/s`;
- שינוי וקטור מרבי לצעד: `0.60m/s`.

Agent 1 search/reacquire לפני Bottom guidance נשאר ללא שינוי.

## דברים שלא שונו

לא שונו:

- PPO checkpoints או מבנה המודלים;
- Action/Observation dimensions;
- `follow_reward_v37.py`;
- `observation_builder.py`;
- `safety_filter.py`;
- `lidar_processor.py`;
- `object_tracker.py`;
- `resnet_yolo_tracker.py`;
- `agent1p2_env.py`;
- `Run_train.py`.

## מה אמור להופיע בריצה

בגובה דומה לצילום, כאשר ה־LIVE BBOX המסונן ממורכז:

```text
BRelGate=CENTER_ONLY
LOCK_PENDING(1/2)
DESCEND_SOFT_CATCHUP_LOCK_ACQUIRED
applied > 0.00
```

בקרבת המטרה:

```text
BRelGate=FINAL
```

ואז `bbox_rel_err` חוזר להיות תנאי לירידה.

ב־CSV נוספו:

```text
bbox_rel_gate_required
bbox_outlier_suppressed
bbox_raw_xyxy
bbox_control_xyxy
parallel_xy_slew_limited
parallel_xy_stable_vx
parallel_xy_stable_vy
```

## מגבלת האימות

בוצעו compilation ובדיקות מבודדות של מנגנוני ה־BBOX, Gate הנחיתה, Z bridge, Down-LiDAR, shared frame ו־XY slew. לא קיימת בסביבה זו ריצת Unreal/AirSim חיה, ולכן אין טענה שהדינמיקה כבר הוכחה בסימולטור. הריצה הראשונה צריכה להיות `Run_diag.py`, לא אימון.
