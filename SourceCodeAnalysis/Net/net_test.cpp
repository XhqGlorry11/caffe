#include<vector>
#include<iostream>
#include<caffe/net.hpp>
using namespace std;
using namespace caffe;

int main(){
    string test_proto("deploy.prototxt");
    //从“deploy.prototxt”文件中解析一个net结构出来
    Net<float> net(test_proto, caffe::TEST);
    //获得net中所有blob的名字
    vector<string> blob_names = net.blob_names();
    for(vector<string>::iterator it=blob_names.begin();it!=blob_names.end();it++){
        cout<<*it<<endl;
    }
    return 0;
}
