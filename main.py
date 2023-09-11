from Utils import InputUtils
from Labs import Lab2

lab_number_input = input('Enter task number: ')

if not InputUtils.is_int(lab_number_input):
    print('Wrong format')
    exit()

lab_number = int(lab_number_input)

match lab_number:
    case 2:
        Lab2.start_lab2()
    case _:
        print(f'Lab{lab_number} doesnt exist')
        exit()
