# Real-time classification of LIDAR data using discrete-time Recurrent Spiking Neural Networks
Due to energy/latency/environmental constraints, Endge AI can benefit from specialized sensors, s.a. LIDAR and optimized learning algorithms. SNN have been shown to have lower energy requirements and to perform well on LIDAR data classification. However, previous work requires the entire frame to be scanned prior to classification, and the usage of large networks.  

We show that a discrete-time Recurrent Spiking Neural Network (RSNN) can efficiently classify LIDAR data, in real-time, throughout the scanning
process. The KITTI 3D Object Detection Benchmark is processed into a labeled object classification dataset, on which we simulate the scanning process and test various optimization techniques and encoding methods.

 ## Process data
  * Download the KITTI 3D Object Detection Benchmark from: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d ;
  * Move the following folders under the same directory: calib, image_2, label_2, velodyne;
  * Run the kitti_utils module with the right command line arguments in order to process the data;
  * [Optional] Analyze the processed data using the analyse_kitti_objects_dataset function from the dataset module.
 ## Run model
  * The mail module trains, tests and saves the module;
  * Use command line arguments to give the right data and result paths, as well as to control the model hyper-papameters.
