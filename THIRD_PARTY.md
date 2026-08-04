# Third-party provenance

| Component | Upstream |
|---|---|
| TRACE method | [*Test-Time Backdoor Detection for Object Detection Models*](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Test-Time_Backdoor_Detection_for_Object_Detection_Models_CVPR_2025_paper.html), CVPR 2025 |
| OGA, ODA, and RMA attack definitions | *BadDet: Backdoor Attacks on Object Detection*, 2022 |
| Detector backend | [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) |
| Dataset and class definitions | [MS COCO](https://cocodataset.org/) |

YOLOv5 and COCO annotations are not vendored. The `test_data/` folders contain
the documented 100 clean and 100 poisoned COCO val2017 images for each attack;
users must follow the applicable COCO terms. Full training data must still be
obtained from the upstream source.
The reproduction guide pins YOLOv5 commit
`b2ffe05569161b7af4e1e3bae617ae25f59d588f`.

The release also includes the experiment backgrounds, NBO image, SSIM reference
crops, and trained checkpoints required by the documented TRACE runs.
