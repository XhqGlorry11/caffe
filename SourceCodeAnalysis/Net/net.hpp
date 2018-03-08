#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer_factory.hpp"

namespace caffe {

// 在图论中，如果一个有向图从任意顶点出发无法经过若干条边回到该点，则这个图是一个有向无环图(DAG图)
/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
// 
template <typename Dtype>
class Net {
 public:
// 显示构造函数，内部调用Init函数
  explicit Net(const NetParameter& param, const Net* root_net = NULL);
  explicit Net(const string& param_file, Phase phase, const Net* root_net = NULL);
// 虚析构函数
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameter.
// Net初始化:创建blobs和layers以搭建整个网络DAG图，以及调用layer的SetUp函数，
// 初始化时也会做另一些记录，例如确认整个网络结构的正确与否等，
// 另外，初始化期间，Net会打印其初始化日志到INFO信息中
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward with the input Blob%s already fed separately.
   *
   * You can get the input blobs using input_blobs().
   */
// 前向传播，以下相关的前向传播函数，内部最终均会调用ForwardFromTo函数
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL);
  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief Run forward using a set of bottom blobs, and return the result.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom, Dtype* loss = NULL);
  /**
   * @brief Run forward using a serialized BlobProtoVector and return the
   *        result as a serialized BlobProtoVector
   */
  string Forward(const string& input_blob_protos, Dtype* loss = NULL);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
// 对Net中的所有diff_数据清零
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
// 反向传播,以下相关的反向传播函数，内部最终均会调用BackwardFromTo函数
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
// 调整layes shape
  void Reshape();

// 前向反向传播
  Dtype ForwardBackward(const vector<Blob<Dtype>* > & bottom) {
    Dtype loss;
    Forward(bottom, &loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
// 更新Net权值和偏置
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
// 共享权值和偏置数据
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
// 从另一个Net拷贝train layers
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
// 从另一个Net拷贝train layers，加载已训练好的模型
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
// 写Net到NetParameter
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
// 写Net weights到HDF5文件
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
// 获得Net名
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
// 获得所有layer名
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
// 获得blob名
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
// 获得blob
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const { return blobs_; }
  /// @brief returns the layers
// 获得layer
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const { return layers_; }
  /// @brief returns the phase: TRAIN or TEST
// 获得Net phase状态：train or test
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
// 获得每一个layer的bottom vector
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const { return bottom_vecs_; }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
// 获得每一个layer的top vector
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const { return top_vecs_; }
  inline const vector<vector<bool> >& bottom_need_backward() const { return bottom_need_backward_; }
  inline const vector<Dtype>& blob_loss_weights() const { return blob_loss_weights_; }
  inline const vector<bool>& layer_need_backward() const { return layer_need_backward_; }
  /// @brief returns the parameters
// 获得各种参数值
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const { return params_; }
  inline const vector<Blob<Dtype>*>& learnable_params() const { return learnable_params_; }
  /// @brief returns the learnable parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  inline const vector<float>& params_weight_decay() const { return params_weight_decay_; }
  inline const vector<bool>& has_params_decay() const { return has_params_decay_; }
  const map<string, int>& param_names_index() const { return param_names_index_; }
  inline const vector<int>& param_owners() const { return param_owners_; }
  /// @brief Input and output blob numbers
// input blob数目
  inline int num_inputs() const { return net_input_blobs_.size(); }
// output blob数目
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const { return net_input_blobs_; }
  inline const vector<Blob<Dtype>*>& output_blobs() const { return net_output_blobs_; }
  inline const vector<int>& input_blob_indices() const { return net_input_blob_indices_; }
  inline const vector<int>& output_blob_indices() const { return net_output_blob_indices_; }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

// 设置是否显示debug info
  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
// 移除指定的layers
  static void FilterNet(const NetParameter& param, NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule, const string& layer_name);

 protected:
  // Helpers for Init.
  /// @brief Append a new input or top blob to the net.
// 追加top blob
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
// 追加bottom blob
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
// 追加blob参数
  void AppendParam(const NetParameter& param, const int layer_id, const int param_id);

// 显示debug info
  /// @brief Helper for displaying debug info in Forward about input Blobs.
  void InputDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

// Caffe中类的成员变量名都带有后缀"_"，这样就容易区分临时变量和类成员变量
  /// @brief The network name
  string name_; // Net名
  /// @brief The phase: TRAIN or TEST
  Phase phase_; // Net Phase状态：train or test
  /// @brief Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_; // layers
  vector<string> layer_names_; // layers名
  map<string, int> layer_names_index_; // layers 索引
  vector<bool> layer_need_backward_; // 指定layers是否需要backward
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype> > > blobs_; // 存储每一个layer产生的中间结果
  vector<string> blob_names_; // blobs名
  map<string, int> blob_names_index_; // blobs 索引
  vector<bool> blob_need_backward_; // 指定blobs是否需要backward
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_; // 存储每一个layer input bottom blobs 指针
  vector<vector<int> > bottom_id_vecs_; // 存储每一个bottom blobs id
  vector<vector<bool> > bottom_need_backward_; // 指定bottom blobs是否需要backward
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_; // 存储每一个layer output top blobs 指针
  vector<vector<int> > top_id_vecs_; // 存储每一个layer output top blobs id
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_; // layer 的loss函数值
  vector<vector<int> > param_id_vecs_; // 
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_; // 
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_; // 是否显示debug info
  /// The root net that actually holds the shared layers in data parallelism
  const Net* const root_net_;

// 禁止使用Net类的拷贝和赋值操作
  DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
