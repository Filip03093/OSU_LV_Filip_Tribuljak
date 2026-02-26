word_counter = {}

with open("song.txt", "r") as file:
    for line in file:
        words = line.split()

        for word in words:
            word = word.strip(".,!?;:()").lower()

            if word in word_counter:
                word_counter[word] += 1
            else:
                word_counter[word] = 1

unique_words = []

for key, value in word_counter.items():
    if value == 1:
        unique_words.append(key)
    
print("Number of unique words: ", len(unique_words))
print("Unique words: ", unique_words)