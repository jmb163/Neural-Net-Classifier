#ifndef UTIL_HPP_
#define UTIL_HPP_
#include<string>
using namespace std;

void strip_char(string&, char*);

void strip_space(string&,bool);

string caps(const string, bool);

int hash_f(const string, int);

string* split(const string, char);

void split_set(string fname, float proportion);

int random_range(const int start, const int end);

int get_index(char*, int, char);

string char_array_to_string(char*, int);

bool is_numeric(string);

void ml_set(string, float, float);

void auto_split_set(string);

#endif /* UTIL_HPP_ */
