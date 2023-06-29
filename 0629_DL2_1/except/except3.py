try:
    fname = input("파일 이름을 입력하세요: ")
    openfile = open(fname, "r")
except IOError:
    print("파일 " +fname+ "을 발견할수 없습니다.")

