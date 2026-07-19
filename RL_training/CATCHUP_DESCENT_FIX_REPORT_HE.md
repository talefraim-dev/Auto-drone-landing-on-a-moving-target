# דוח תיקון — Agent 2 אינו מנמיך בזמן Predictive Catch-up

## הכשל שאומת

בצילום הופיע:

```text
Z_CTRL=HOLD_PREDICTIVE_CATCHUP raw=+0.93 down=+0.80 applied=+0.00
LOCK_PENDING(0/2)
CATCH=1
```

כלומר PPO ביקש ירידה של `0.80 m/s`, אך ה־gate הדטרמיניסטי איפס אותה.
במקביל, עצם היות `CATCH=1` איפס בכל צעד את רצף ה־Landing Lock, גם כאשר
המטרה הייתה LIVE, בעלת similarity גבוהה ויציבה ליד מרכז התמונה.

נמצא גם קונפליקט שני: לאחר פעימת Agent 2, שכבת ה־command bridge של Agent 1
שלחה פקודת המשך עם `vz=0`. לכן גם פתיחת ה־gate לבדה הייתה משאירה את הירידה
כפעימה קצרה שכמעט אינה נראית במחזור ראייה ארוך.

## התיקון

### 1. הפרדת Horizontal Catch-up מ־Vertical Unsafe

`predictive_catchup_active` אינו עוד וטו אנכי אוטומטי. בזמן catch-up מותר
לבנות Landing Lock רק כאשר מתקיימים יחד:

- Bottom LIVE match מאומת.
- similarity מעל סף הירידה.
- שגיאת מרכז ו־bbox בתוך ספי הכניסה ל־Landing Lock.
- מדידת התנועה תקפה מבחינת attitude.
- התחזית מבוססת LIVE ולא PRED בלבד.
- מרכז התמונה החזוי קטן מ־`0.18`.
- מהירות התמונה קטנה מ־`0.22 /s`.
- מהירות ההתרחקות הרדיאלית בתמונה קטנה מ־`0.06 /s`.

PRED-only, זהות לא מאומתת, תחזית מתבדרת, attitude לא תקף או סטייה גדולה
עדיין חוסמים ירידה מיד.

### 2. Landing Lock בזמן Catch-up בטוח

- פריים בטוח ראשון: `ALIGN_LOCK_PENDING_SOFT_CATCHUP`.
- פריים בטוח שני: `DESCEND_SOFT_CATCHUP_LOCK_ACQUIRED`.
- לאחר מכן: `DESCEND_SOFT_CATCHUP_LANDING_LOCK`.

אין יותר איפוס קבוע ל־`0/2` כאשר התמונה החיה בטוחה.

### 3. ירידה מוגבלת

- Catch-up בטוח: עד `0.28 m/s` NED-Z חיובי.
- מתחת ל־`1.0 m` מעל המטרה: עד `0.12 m/s`.
- לאחר שחרור catch-up, מסלול הירידה הרגיל נשאר ללא שינוי.

### 4. תיקון קונפליקט ה־command bridge

הגישור מרענן כעת גם Z חיובי שכבר אושר, במקום לאפס אותו מיד אחרי הפעימה.
הוא אינו משתמש ב־Z הגולמי של Agent 2 אלא ב־Z שנשלח בפועל לאחר safety:

- cap גישור אנכי כללי: `0.35 m/s`.
- Down-LiDAR limit נשמר, למשל `0.12 m/s`.
- Down-LiDAR block או hard safety מחזירים Z לאפס.
- reacquire climb שלילי לעולם אינו נשמר.
- Pitch/Roll מעל מעטפת ה־bridge מבטלים גם XY וגם Z של הגישור.

### 5. דיאגנוסטיקה

נוספו השדות:

- `vertical_speed_limit_mps`
- `soft_catchup_descent_active`
- `bridge_vz`
- `parallel_z_blocked_by_safety`
- `parallel_z_limited_by_safety`

## רכיבים שלא שונו

לא שונו:

- מבנה PPO או ממדי Observation/Action.
- `follow_reward_v37.py`.
- `observation_builder.py`.
- `safety_filter.py` עצמו.
- `lidar_processor.py`.
- `object_tracker.py` ו־`resnet_yolo_tracker.py`.
- קובצי המודל וה־checkpoints.

## ולידציה

עברו:

- 6 בדיקות ייעודיות ל־catch-up descent.
- בדיקות command bridge, Z cap, hard attitude, hard safety, Down-LiDAR
  limit/block ואיסור persistent reacquire climb.
- בדיקות Agent-2 horizontal/vertical/safe-training.
- בדיקות shared bottom perception וזהות המטרה.
- בדיקות hash של רכיבי הליבה.
- AST parsing ו־compileall לכל 155 קובצי Python.

שתי בדיקות collision ישנות נכשלות גם ב־TRUE_ROOT_FIX המקורי ואינן קשורות
לעדכון: אחת מקובעת ל־`TOTAL_AGENT2_TIMESTEPS=10_240`, והשנייה מקובעת ללוגיקת
collision-age ישנה. שתי בדיקות נוספות דורשות `gymnasium`/`cosysairsim` שאינם
מותקנים בסביבת הבדיקה.

לא בוצעה טיסת AirSim חיה כאן. יש להריץ `Run_diag.py` לפני אימון נוסף.
