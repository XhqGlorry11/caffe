# Train LeNet using pycaffe interface.
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('/home/pc-201/Desktop/solver.prototxt')
solver.solve()


And the solver.prototxt looks like follows:
train_net:"/home/pc-201/Desktop/lenet_train_test.prototxt"
test_net:"/home/pc-201/Desktop/lenet_train_test.prototxt"
tess_interval:500
base_lr:0.01
momentum:0.9
weight_decay:0.0005
lr_policy:"inv"
gamma:0.0001
power:0.75
display:100
max_iter:10000
snapshot:5000
snapshot_prefix:"/home/pc-201/"
