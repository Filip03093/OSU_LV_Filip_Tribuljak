ham_word_count = 0
spam_word_count = 0
ham_count = 0
spam_count = 0
spam_exclamation_count = 0

with open("SMSSpamCollection.txt", "r") as file:
    for line in file:
        line = line.strip()
        
        if line.startswith("ham"):
            ham_count += 1
            message = line[4:]
            ham_word_count += len(message.split())
        
        elif line.startswith("spam"):
            spam_count += 1
            message = line[5:]
            spam_word_count += len(message.split())
            
            if message.endswith("!"):
                spam_exclamation_count += 1

average_ham = ham_word_count / ham_count
average_spam = spam_word_count / spam_count

print("Average number of words in ham messages:", average_ham)
print("Average number of words in spam messages:", average_spam)
print("Number of spam messages ending with '!':", spam_exclamation_count)