import numpy as np

def get_config():
    """הגדרות פרמטרים לאימון גנרי וחסין (Robust)"""
    return {
        "model_name": "drone_generic_landing_v1",
        "total_timesteps": 500000,  # הגדלה משמעותית כדי ללמוד הכללה תחת שינויי תאורה
        "learning_rate": 2e-4,      # קצב למידה מעט נמוך יותר ליציבות גבוהה
        "n_steps": 2048,
        "batch_size": 128           # באץ' גדול יותר עוזר למודל להבין את הממוצע של הרעש
    }

def compute_reward(obs, action, done):
    # הסביבה (droneEnv) כבר מחשבת את הפרס, פונקציה זו נשארת לתאימות
    return 0