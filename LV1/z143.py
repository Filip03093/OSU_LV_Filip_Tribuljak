empty_list = []
while True:
    user_input = input("Enter a number or 'Done' to finish: ")

    if user_input == 'Done':
        break

    try:
        number = int(user_input)
        empty_list.append(number)
    except ValueError:
        print("Input should be only numbers")

print("Number count: ", len(empty_list))
print("Average: ", float(sum(empty_list)/len(empty_list)))
print("Min: ", min(empty_list))
print("Max: ", max(empty_list))
empty_list.sort()
print(empty_list)