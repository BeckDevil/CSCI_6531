#include <iostream>
#include <cstdio>
#include <cstring>
float char_freq[26]= {0.080, 0.015, 0.030, 0.040, 0.130, 0.020, 0.015,
					  0.060, 0.065, 0.005, 0.005, 0.035, 0.030,
					  0.070, 0.080, 0.020, 0.002, 0.065, 0.060, 
					  0.090, 0.030, 0.010, 0.015, 0.005, 0.020, 0.002						
					 };

int char_to_idx(char c)
{
	int x = int(c-'A');
	return x;
}

char idx_to_char(int x)
{
	char c = char(x+'A');
	return c;
}

int main(int argc, char* argv[])
{
	std::string cipher = "TEBKFKQEBZLROPBLCERJXKBSBKQP";
	float score;
	char plaintext[29];
	int frequency[26];
	for (int i = 0; i < 26; ++i){
		frequency[i] = 0;
	}
	for (int i = 0; i < 28; i++){
		int id = char_to_idx(char(cipher[i]));
//		printf("%d ",id);
		frequency[id] += 1;
	}
	printf("\n");
	for (int i = 0; i < 26; i++){
//		printf("Freq of letter %c is %d\n", idx_to_char(i), frequency[i]);
	}

	plaintext[28] = '\0';
	for(int key = 0; key <= 25; ++key){
		printf("Key: %d, ", key);
		score = 0.0;
		int idx;
		for(int i = 0; i <= 25; ++i){
			idx = (26+i-key)%26;
			score += frequency[i]*char_freq[idx];
		}
		printf("Correlation: %f, ", score);
		for(int i = 0; i < 28; i++){
			idx = char_to_idx(cipher[i]);
			idx = (26+idx - key)%26;
			plaintext[i] = idx_to_char(idx);
		}
		printf("Plaintext: %s\n", plaintext);
	}
	int id = 4;
/*
	char op = idx_to_char(id);
	int idx = char_to_idx(x);

	printf("char %c to id %d\n", x, idx);
	printf("id %d to char %c\n", id, op);
*/	
	return 0;

}
