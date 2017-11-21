#include<iostream>
#include<fstream>
#include<sstream>
#include"util.hpp"

using namespace std;

int main(int argc, char** argv)
{
    ifstream in(argv[1]);
    if(!in||in.fail())
    {
        cerr<<"Read error\n";
        exit(-1);
    }
    double* averages = NULL;
    int num_columns;
    int num_instances = 0;
    string hold;
    
    stringstream ss;
    while(getline(in, hold))
    {
        strip_char(hold, "\n");
        string* data = split(hold, ',');
        if(averages == NULL)
        {
            //we have to figure out how many columns are in the data
            ss<<data[0];
            ss>>num_columns;
            num_columns--;
            averages = new double[num_columns];
            for(int j =0; j<num_columns; ++j)
            {
                averages[j] = 0;
            }
        }
        for(int i=0; i<num_columns; ++i)
        {
            if(is_numeric(data[i + 1]))
            {
                ss.str(std::string());
                ss.clear();
                double dat;
                ss<<data[i + 1];
                ss>>dat;
                cout<<dat<<",";
                //cout<<data[i+1]<<",";
                averages[i] = averages[i] + dat;
            }
            else
            {
                cout<<"is not numeric"<<endl;
            }
        }
        cout<<endl;
        num_instances++;
    }
    cout<<"num_instances is: "<<num_instances<<endl;
    cout<<"num_columns is: "<<num_columns<<endl;
    for(int i=0; i<num_columns; ++i)
    {
        averages[i] = averages[i] / (double)num_instances;
        cout<<"average for column "<< i <<" is: "<<averages[i]<<endl;
    }
    in.close();
    
    ifstream in2(argv[1]);
    if(!in2||in2.fail())
    {
        cerr<<"Read error\n";
        exit(-1);
    }
    ofstream out("processed");
    if(!out||out.fail())
    {
        cerr<<"Read error\n";
        exit(-1);
    }
    
    while(getline(in2, hold))
    {
        strip_char(hold, "\n");
        string* data = split(hold, ',');
        
        for(int i = 0; i<num_columns; ++i)
        {
            if(!is_numeric(data[i + 1]))
            {
                out<<averages[i];
            }
            else
            {
                out<<data[i + 1];
            }
            
            if(i != (num_columns - 1))
            {
                out<<",";
            }
        }
        out<<endl;
    }
    in2.close();
    out.close();
    return 0;
}

//int main(int argc, char** argv)
//{
//    ifstream in(argv[1]);
//    if(!in||in.fail())
//    {
//        cerr<<"could not open input stream"<<endl;
//        exit(-1);
//    }
//    string hold;
//    int instances=0;
//    
//    getline(in,hold);
//    strip_char(hold, "\n");
//    string* column=split(hold, ',');
//    int num_elements;
//    stringstream ss;
//    ss<<column[0];
//    ss>>num_elements;
//    num_elements--;
//    double* averages=new double[num_elements];
//    for(int i=0; i<num_elements+1; ++i)
//    {
//        if(is_numeric(column[i+1]))
//        {
//            ss.str(std::string());
//            ss<<column[i+1];
//            double value;
//            ss>>value;
//            averages[i]+=value;
//        }
//    }
//    instances++;
//    
//    while(getline(in, hold))
//    {
//        for(int i=0; i<num_elements+1; ++i)
//        {
//            if(is_numeric(column[i+1]))
//            {
//                ss.str(std::string());
//                ss<<column[i+1];
//                double value;
//                ss>>value;
//                averages[i]+=value;
//            }
//        }
//        instances++;
//    }
//    
//    for(int i = 0; i<num_elements; ++i)
//    {
//        averages[i]=averages[i]/(double)instances;
//    }
//    //now that we have the averages, we can sanitize the dataset
//    in.clear();
//    in.seekg(0, ios::beg);
//    while(getline(in, hold))
//    {
//        strip_char(hold, "\n");
//        string* column=split(hold,',');
//        int i=0;
//        for(i; i<num_elements-1; ++i)
//        {
//            if(column[i+1]=="?")
//            {
//                cout<<averages[i]<<",";
//            }
//            else
//            {
//                cout<<column[i+1]<<",";
//            }
//        }
//        if(column[i+1]=="?")
//        {
//            cout<<averages[num_elements-1];
//        }
//        else
//        {
//            cout<<column[num_elements-1];
//        }
//
//        cout<<endl;
//    }
//    
//    return 0;
//}
