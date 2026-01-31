# COLET: A Dataset for Cognitive Workload Estimation based on Eye-tracking

- [Link to Paper](https://www.sciencedirect.com/science/article/pii/S0169260722003716)
- [Link to Dataset](https://zenodo.org/records/7766785)

Database including eye movements from 47 participants as they solved puzzles involving visual search tasks of varying complexity and duration.

Participants rated their performance based on [NASA's RTLX index scale](https://www.ejbi.org/scholarly-articles/nasa-rtlx-as-a-novel-assessment-tool-for-determining-cognitive-load-and-user-acceptance-of-expert-and-userbased-usabilityevaluatio-5988.html).

## Sensor Information

This study used [Pupil Labs' "Pupil Core" eye-tracker](https://pupil-labs.com/products/core). Recordings were binocular with 240 Hz sampling frequency, 0.60 degrees accuracy and 0.02 precision.

- On-body Eye Tracker
- 240 Hz SR


## Data Format

We have post-processed the original data, so that it is quite clean.

The files we have uploaded are:
1. `participant{i}/` directory: Folders containing the `gaze.csv`, `pupil.csv`, `blinks.csv` and `annotation.json` for each of 4 tasks under respective `task{j}/` subdirectories.
2. `subject_info.csv`: General information about the subjects.
3. `images/` folder: A folder containing the 21 images shown.


### Participant Data

```bash
colet
├───participant{i}
│   ├───task{j}
│   └───task{j}
└───participant{i}
    ├───task{j}
    └───task{j}
```

We have 47 participants in this study, hence `i ∈ [0,46]` which is the participant id.

We have only 4 tasks, hence `j ∈ [0,3]` which is the task number (refer to publication for information on these tasks).

Each task compiles the following data:

#### Gaze (`gaze.csv`)

This collects gaze-related metrics recorded from the eye tracker for each of the 4 total tasks.

1. `gaze_timestamp`
2. `world_index`
3. `confidence`
4. `norm_pos_x`, `norm_pos_y`
5. `base_data`
6. `gaze_point_3d_x`, `gaze_point_3d_y`, `gaze_point_3d_z`
7. `eye_center0_3d_x`, `eye_center0_3d_y`, `eye_center0_3d_z`
8. `gaze_normal0_x`, `gaze_normal0_y`, `gaze_normal0_z`
9. `eye_center1_3d_x`, `eye_center1_3d_y`, `eye_center1_3d_z`
10. `gaze_normal1_x`, `gaze_normal1_y`, `gaze_normal1_z`

#### Pupilometry (`pupil.csv`)

This collects pupil-related metrics recorded from the eye tracker for each of the 4 total tasks.

1. `pupil_timestamp`
2. `world_index`
3. `eye_id`
4. `confidence`
5. `norm_pos_x`, `norm_pos_y`
6. `diameter`
7. `method`
8. `ellipse_center_x`, `ellipse_center_y`
9. `ellipse_axis_a`, `ellipse_axis_b`
10. `ellipse_angle`
11. `diameter_3d`
12. `model_confidence`
13. `model_id`
14. `sphere_center_x`, `sphere_center_y`, `sphere_center_z`
15. `sphere_radius`
16. `circle_3d_center_x`, `circle_3d_center_y`, `circle_3d_center_z`
17. `circle_3d_normal_x`, `circle_3d_normal_y`, `circle_3d_normal_z`
18. `circle_3d_radius`
19. `theta`
20. `phi`
21. `projected_sphere_center_x`, `projected_sphere_center_y`
22. `projected_sphere_axis_a`, `projected_sphere_axis_b`
23. `projected_sphere_angle`

#### Blinks (`blinks.csv`)

This collects blink-related metrics recorded from the eye tracker for each of the 4 total tasks.

1. `id`
2. `start_timestamp`, `duration`, `end_timestamp`
3. `start_frame_index`, `index`, `end_frame_index`
4. `confidence`
5. `filter_response`
6. `base_data`

#### Annotation (`annotation.json`)

This collects the NASA RTLX scores for each of the 4 total tasks.

For more info regarding the recordings from Pupil Core visit [Pupil Capture webpage](https://docs.pupil-labs.com/core/software/pupil-capture/).



### Subject Info (`subject_info.csv`)

This is a table collecting the following general subject information:

1. Visual acuity: measured binocularly in distance of 0.80cm (logMAR)
2. Gender ('F': Female, 'M': Male)  
3. Age (years)
4. Education level (years)



### `images/` folder

Folder containing the images used in the experiment. Mapping to task is as follows:

```
1 --> bowling_balls
2 --> candles
3 --> chandelier
4 --> classroom
5 --> garage
6 --> handles
7 --> kitchen
8 --> light
9 --> paintings_1
10 --> paintings_2
11 --> pc_screens
12 --> pillows
13 --> poof
14 --> pool
15 --> seats_1
16 --> seats_2
17 --> shoes
18 --> students
19 --> towels
20 --> water
21 --> windows
```

## Citation

```bibtex
@article{ktistakis2022colet,
    title = {COLET: A dataset for COgnitive workLoad estimation based on eye-tracking},
    journal = {Computer Methods and Programs in Biomedicine},
    volume = {224},
    pages = {106989},
    year = {2022},
    issn = {0169-2607},
    doi = {https://doi.org/10.1016/j.cmpb.2022.106989},
    url = {https://www.sciencedirect.com/science/article/pii/S0169260722003716},
    author = {Emmanouil Ktistakis and Vasileios Skaramagkas and Dimitris Manousos and Nikolaos S. Tachos and Evanthia Tripoliti and Dimitrios I. Fotiadis and Manolis Tsiknakis},
    keywords = {Cognitive workload, Workload classification, Eye movements, Machine learning, Eye-tracking, Affective computing}
}
```
