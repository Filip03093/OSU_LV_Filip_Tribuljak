try:
    grade = float(input("Grade between 0.0-1.0: "))
    if grade < 0 or grade > 1:
        raise Exception("Grade must be between 0.0 and 1.0")
    if grade < 0.6:
        print('F')
    elif grade < 0.7:
        print('D')
    elif grade < 0.8:
        print('C')
    elif grade < 0.9:
        print('B')
    else:
        print('A')
except Exception as e:
    print(e)