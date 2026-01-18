import cv2
import numpy as np
import torch
from ultralytics import YOLO
import cosysairsim as airsim

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    model = YOLO("yolo11n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    while True:
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        if not responses or not responses[0].image_data_uint8:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img.reshape(responses[0].height, responses[0].width, 3)
            frame = cv2.resize(frame, (640, 360))

        res = model.predict(frame, conf=0.10, verbose=False)[0]

        # draw ALL boxes
        if res.boxes:
            for box in res.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
                cv2.putText(frame, f"{cls}:{conf:.2f}", (b[0], max(0, b[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("YOLO RAW BOXES (conf=0.10)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
