# Paper Library

This folder now contains the main papers used as evidence for feature extraction and for pipeline / model design.

## Biomedical and feature-extraction evidence

| Paper | Why it matters for this project | Local PDF |
| :--- | :--- | :--- |
| PERCLOS and alertness measurement report | Foundational reference for eye-closure / alertness logic. | [PDF](./biomedical/perclos_1998_fhwa_mcrt_98_006.pdf) |
| Eye blink detection using facial landmarks | Supports EAR-style blink / eyelid-closure features. | [PDF](./soukupova_cech_2016_eye_blink_detection.pdf) |
| Detecting Driver Drowsiness Based on Sensors: A Review | Early review covering behavioral, physiological, and vehicle-based drowsiness cues. | [PDF](./biomedical/detecting_driver_drowsiness_based_on_sensors_review_2012.pdf) |
| Cues of fatigue: effects of sleep deprivation on facial appearance | Strong biomedical evidence that sleep loss changes eyes, mouth, and skin. | [PDF](./biomedical/cues_of_fatigue_sleep_deprivation_2013.pdf) |
| Feature selection for driving fatigue characterization and detection using visual- and signal-based sensors | Useful for justifying a mixed visual + signal feature set. | [PDF](./biomedical/feature_selection_driving_fatigue_characterization_detection_visual_signal_sensors_2018.pdf) |
| Dynamic Visual Measurement of Driver Eye Movements | Supports eye-movement dynamics as fatigue / vigilance signals. | [PDF](./biomedical/dynamic_visual_measurement_driver_eye_movements_2019.pdf) |
| Driver Fatigue Detection Based on RCAN and Head Pose Estimation | Strong support for eye state, mouth state, PERCLOS, and head pose. | [PDF](./biomedical/driver_fatigue_detection_rcan_head_pose_2021.pdf) |
| Driver’s Head Pose and Gaze Zone Estimation Based on Multi-Zone Templates Registration and Multi-Frame Point Cloud Fusion | Supports head pose and gaze-zone features as visual attention cues. | [PDF](./biomedical/driver_head_pose_gaze_zone_estimation_2022.pdf) |
| Real-Time Machine Learning-Based Driver Drowsiness Detection Using Visual Features | Very close to your pipeline: MediaPipe / landmarks / EAR / MAR / head pose. | [PDF](./biomedical/real_time_machine_learning_based_driver_drowsiness_detection_visual_features_2023.pdf) |
| A Review of Recent Developments in Driver Drowsiness Detection Systems | Recent survey of current drowsiness detection methods and tradeoffs. | [PDF](./biomedical/a_review_of_recent_developments_driver_drowsiness_detection_systems_2022.pdf) |
| Association of Visual-Based Signals with EEG Patterns in Enhancing the Drowsiness Detection in Drivers with Obstructive Sleep Apnea | Biomedical evidence that visual cues correlate with EEG patterns in OSA drivers. | [PDF](./biomedical/association_visual_based_signals_eeg_osa_drowsiness_2024.pdf) |
| A Review of Driver Drowsiness Detection Systems: Techniques, Advantages and Limitations | Broad review that supports the overall design choice of hybrid drowsiness systems. | [PDF](./biomedical/a_review_of_driver_drowsiness_detection_systems_techniques_advantages_limitations_2022.pdf) |
| Head pose and visual attention mapping paper | Supports the use of head pose as a visual-attention signal. | [PDF](./biomedical/head_pose_attention_2017.pdf) |

## Pipeline and model references

| Paper | Why it matters for pipeline / model design | Local PDF |
| :--- | :--- | :--- |
| OpenFace 2.0: Facial Behavior Analysis Toolkit | Core face-behavior / AU / gaze / pose feature extractor reference. | [PDF](./pipeline/openface_2_0_facial_behavior_analysis_toolkit_2018.pdf) |
| MediaPipe: A Framework for Building Perception Pipelines | Good reference for the perception-pipeline layer in production. | [PDF](./pipeline/mediapipe_framework_perception_pipelines_2019.pdf) |
| DAiSEE: Towards User Engagement Recognition in the Wild | Base dataset / engagement benchmark reference. | [PDF](./pipeline/daisee_2016_towards_user_engagement_recognition_in_the_wild.pdf) |
| Copur thesis 2021 | Very close to your OpenFace/OpenPose feature aggregation and Bi-LSTM setup. | [PDF](./copur_thesis_2021.pdf) |
| Santoni et al. 2023 (OpenFace 709D -> PCA/SVD -> CNN) | Important SOTA reference for feature reduction + CNN on DAiSEE. | [PDF](./santoni_ijacsa_2023.pdf) |
| Improving state-of-the-art in Detecting Student Engagement with Resnet and TCN Hybrid Network | Direct model-design reference for temporal / hybrid network ideas. | [PDF](./pipeline/improving_state_of_art_student_engagement_resnet_tcn_2021.pdf) |
| A General Model for Detecting Learner Engagement: Implementation and Evaluation | Useful for engagement-model framing and evaluation structure. | [PDF](./pipeline/a_general_model_detecting_learner_engagement_2024.pdf) |
| MultiMediate'23: Engagement Estimation and Bodily Behaviour Recognition in Social Interactions | Helpful for sequence modeling and multimodal engagement reasoning. | [PDF](./pipeline/multimediate23_engagement_estimation_social_interactions_2023.pdf) |

## Notes

- The biomedical set is the main evidence base for why EAR, MAR, head pose, gaze, blink, and facial dynamics are reasonable features.
- The pipeline/model set is the main evidence base for how to structure the extractor -> sequence -> classifier flow.
- Files are stored locally and ready for direct use in reports, slides, or deployment docs.
