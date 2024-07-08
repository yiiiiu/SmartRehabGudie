class a():
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def run(self):
        pass

    def start(self):
        self.run()

class b(a):
    def __init__(self, num_array) -> None:
        super().__init__()
        self.running = True
        self.num= num_array

    def run(self):
        count = 0
        while self.running:
            count += 1
            print(f'当前计数:{count}')
            if count == 9:
                print('切换下一个动作：'+ self.num[0] )
                self.num = self.num[1:]
                print(self.num)
                if not self.num:
                    print("切换结束！")
                    return
                break
        self.cycle()

    def cycle(self):
        self.run()



if __name__ == '__main__':
    num_array = ['0', '1', '2', '3', '4']
    _b = b(num_array)
    _b.start()

