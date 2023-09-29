from Utils import InputUtils
from Labs import Lab1, Lab2, Lab3, Lab4

lab_number_input = input('Enter lab number: ')

if not InputUtils.is_int(lab_number_input):
    print('Wrong format')
    exit()

lab_number = int(lab_number_input)

match lab_number:
    case 1:
        Lab1.start_point()
    case 2:
        Lab2.start_point()
    case 3:
        Lab3.start_point()
    case 4:
        Lab4.start_point()
    case _:
        print(f'Lab{lab_number} doesnt exist')
        exit()
