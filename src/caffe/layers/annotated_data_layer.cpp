#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>
#include <fstream>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"


namespace caffe {

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  // Read a data point, and use it to initialize the top blob.
  AnnotatedDatum& anno_datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  //LOG(INFO) << "output data size: " << top[0]->num() << ","
  //    << top[0]->channels() << "," << top[0]->height() << ","
  //    << top[0]->width();
  // label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  AnnotatedDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;

	vector<float> origin_coord(4, 0);
	float prob_thrshld = anno_data_param.part_sampler_prob();
	bool is_sampler_part = false;
	if (prob_thrshld > 1e-6){
		float prob_partsample;
		caffe_rng_uniform(1, 0.f, 1.f, &prob_partsample);
		if (prob_partsample < prob_thrshld)
			is_sampler_part = true;
	}

    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
		/*if (is_sampler_part)
        	this->data_transformer_->ExpandImage(distort_datum, expand_datum, &origin_coord);
		else*/
			this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
		/*if (is_sampler_part)
			this->data_transformer_->ExpandImage(distort_datum, expand_datum, &origin_coord);
		else*/
			this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }

	//LOG(INFO) << "n xmin: " << origin_coord[0];
	//LOG(INFO) << "n xmax: " << origin_coord[1];
	//LOG(INFO) << "n width: " << origin_coord[2];
	//LOG(INFO) << "n height: " << origin_coord[3];

    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
	  if (is_sampler_part){
		  vector<NormalizedBBox> sampled_part_bboxes;
		  GenerateBatchSamples_Part(*expand_datum, batch_samplers_, &sampled_part_bboxes);// , origin_coord);
		  if (sampled_part_bboxes.size() > 0){
			  //LOG(INFO) << "croped part" << sampled_part_bboxes.size();

			  /*float origin_width = origin_coord[4];
			  float origin_height = origin_coord[5];
			  float expand_width = origin_coord[6];
			  float expand_height = origin_coord[7];
			  float w_off = origin_coord[8];
			  float h_off = origin_coord[9];*/

			  //float part_xmin = sampled_part_bboxes[0].xmin() * origin_coord[6];
			  //float part_ymin = sampled_part_bboxes[0].ymin() * origin_coord[7];
			  //float part_xmax = sampled_part_bboxes[0].xmax() * origin_coord[6];
			  //float part_ymax = sampled_part_bboxes[0].ymax() * origin_coord[7];

			  /*float part_xmin = std::max(0.f, sampled_part_bboxes[0].xmin() * expand_width - w_off);
			  float part_ymin = std::max(0.f, sampled_part_bboxes[0].ymin() * expand_height - h_off);
			  float part_xmax = std::min(w_off + origin_coord[4], sampled_part_bboxes[0].xmax() * expand_width - w_off);
			  float part_ymax = std::min(h_off + origin_coord[5], sampled_part_bboxes[0].ymax() * expand_height - h_off);*/

			  /*std::ofstream file_save;
			  file_save.open("cropped_img.txt", std::ofstream::app);
			  LOG(INFO) << "origin_width: " << origin_width << " origin_height: " << origin_height 
						<< " expand_width: " << expand_width << " expand_height: " << expand_height;
			  LOG(INFO) << "cropped coordinates: " << part_xmin << " " << part_ymin << " " << part_xmax << " " << part_ymax;
			  file_save << part_xmin << " " << part_ymin << " " << part_xmax << " " << part_ymax << " "
						<< w_off << " " << h_off << " " << expand_width << " " << expand_height << "\n";
			  file_save.close();*/


			  //LOG(INFO) << "sampled part bboxes ------";
			  sampled_datum = new AnnotatedDatum();
			  this->data_transformer_->CropImage(*expand_datum,
				  sampled_part_bboxes[0],
				  sampled_datum);
			  has_sampled = true;
		  }
		  else {
			  LOG(INFO) << "sampled part bboxes ------ not success";
			   vector<NormalizedBBox> sampled_bboxes;
			  GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
			  if (sampled_bboxes.size() > 0){
				  //LOG(INFO) << "sampled normal bboxes";
				  int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
				  sampled_datum = new AnnotatedDatum();
				  this->data_transformer_->CropImage(*expand_datum,
					  sampled_bboxes[rand_idx],
					  sampled_datum);
				  has_sampled = true;
			  }
			  else {
				  sampled_datum = expand_datum;
			  }
		  }
	  }
	  else{
		  vector<NormalizedBBox> sampled_bboxes;
		  GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
		  if (sampled_bboxes.size() > 0){
			  //LOG(INFO) << "sampled normal bboxes";
			  int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
			  sampled_datum = new AnnotatedDatum();
			  this->data_transformer_->CropImage(*expand_datum,
				  sampled_bboxes[rand_idx],
				  sampled_datum);
			  has_sampled = true;
		  }
		  else {
			  sampled_datum = expand_datum;
		  }
	  }

    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        this->data_transformer_->Transform(*sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
		  //LOG(INFO) << "num_bboxes: " << num_bboxes;
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
  }

  //std::ofstream anno_save;
  //anno_save.open("anno_save.txt", std::ofstream::app);

  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
		//LOG(INFO) << "Batch_size " << batch_size;
		//anno_save << "Batch_size " << batch_size << " ";
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
		  //LOG(INFO) << "anno_vec.size() " << anno_vec.size();
		  //anno_save << "batch: " << item_id << " ";
          for (int g = 0; g < anno_vec.size(); ++g) {
			//anno_save << "anno_vec:  " << g << " ";
            const AnnotationGroup& anno_group = anno_vec[g];
			//LOG(INFO) << "anno_group.annotation_size() " << anno_group.annotation_size();
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
			  //anno_save << "anno_group: " << a << " ";
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
			  //anno_save << bbox.xmin() << " " << bbox.ymin() << " " << bbox.xmax() << " " << bbox.ymax() << " ";
			  //LOG(INFO) << bbox.xmin() << " " << bbox.ymin() << " " << bbox.xmax() << " " << bbox.ymax() << " ";
            }
          }
		  //anno_save << "\n";
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  //anno_save.close();

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
