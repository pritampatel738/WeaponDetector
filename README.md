<h2> Weapon detector capable of detecting Rifles and Handguns.</h2>

<h2> Using Pretrained Model </h2>
(1) Create a folder named "MWeapon" or anything in C: directory.

(2) First of all you need to download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file.
After extracting the files rename "models-master" to "models" and copy models to MWeapon directory.

(3) Append it to the path ...
Go to your anaconda prompt and inside C:> type below command ...... 
C:\>set PYTHONPATH="C:\MWeapon\models;C:\MWeapon\models\research;C:\MWeapon\models\research\slim"

(4) Compile all your protos in models/research/object_detection folder. Run the below command.
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto

if you are having an error , then specify the explicit path to protoc i.e, replace first protoc to path to 
protoc. e.g: (C:/MWeapon/bin/protoc) 

(5) Now go to given google drive link and download 4 files ..
      https://drive.google.com/open?id=1hfCql15CAGWT8zWF8vNWB8rfeRXgtdEl
      (a) training
      (b) inference_graph
      (c) train
      (d) test
Extract all the files on your desktop.

(6) Paste train and test folders inside "images" directory as it is in repo. Now you have 4 files in "images" 
folder , two train and test files and two corresponding train and test labels.

(7) Now put training and inference_graph folder in GunDetector.
(8) Now your GunDetector has 3 folders(images,training,inference_graph) and some python files.
(9) Now copy all the contents inside GunDetector(all folders and python files) and put them into models
directory that you have downloaded in step1.

(10) Congrats, now you're ready to go. just run python Object_detection_image.py or Object_detection_video1.py to
run the pretrained model and specify the path in case of image and video. If you want to use webcam then run 
Object_detection_webcam.py .

<h2> References </h2>
<ul>
  <li> Edje Electronics (Evan)  </li>
  https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
  <li> Dat Tran / raccoon_datasets </li>
   https://github.com/datitran/raccoon_dataset
 </ul>
