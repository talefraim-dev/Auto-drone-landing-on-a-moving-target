import torch
import numpy  # ייבוא נחוץ לתיקון האבטחה

file_path = "SiamMask/experiments/siammask_sharp/SiamMask_VOT.pth"

try:
    # הוספת numpy.core.multiarray.scalar לרשימת המותרים או ביטול weights_only
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    print("✅ הקובץ נטען בהצלחה!")

    # בדיקה שהתוכן אכן שם
    if isinstance(checkpoint, dict):
        print(f"המפתח 'state_dict' נמצא: {'state_dict' in checkpoint}")
        print(f"נמצאו {len(checkpoint.get('state_dict', checkpoint))} שכבות.")
except Exception as e:
    print(f"❌ שגיאה בטעינה: {e}")