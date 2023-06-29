while True:
    try:
        num = input("숫자를 입력하시오. : ")
        num = int(num)
        break
    except ValueError:
        print("정수가 아닙니다. 다시 입력하시오. ")
print("정수 입력이 성공했습니다.")

