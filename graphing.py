import pandas as pd
import matplotlib.pyplot as plt

def graphing(result_file_path):
    def str2num_list(s):
        return [float(i) for i in s.split('|')]
    
    def draw_acc_loss(titles, train_matric, dev_matric):
        plt.figure(figsize=(32,16))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        for ix, tit, t_mat, d_mat in zip(range(len(titles)), titles, train_matric, dev_matric):    
            t_mat = str2num_list(t_mat)
            d_mat = str2num_list(d_mat)
            plt.subplot(5,3,ix+1)
            plt.grid()
            plt.title(tit)
            plt.xlabel('epoch')
            plt.plot(range(len(t_mat)), t_mat, label='train')
            plt.plot(range(len(d_mat)), d_mat, label='dev')
            plt.legend()
        plt.savefig('acc_loss.jpg')
        plt.show()

    def draw_test_matrics(titles, acc, loss):
        plt.figure(figsize=(32,16))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
        plt.subplot(121)
        plt.grid()
        plt.title('Acc')
        rects = plt.bar(titles, acc)
        plt.xlabel('Model with different modules')
        plt.ylabel('Acc')
        plt.xticks(titles, titles, rotation=30)
        for rect in rects:  
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(round(height, 2)), size=15, ha='center', va='bottom')

        plt.subplot(122)
        plt.grid()
        plt.title('Loss')
        rects = plt.bar(titles, loss)
        plt.xlabel('Model with different modules')
        plt.ylabel('Loss')
        plt.xticks(titles, titles, rotation=30)
        for rect in rects:  
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(round(height, 2)), size=15, ha='center', va='bottom')
        plt.savefig('test_acc_loss.jpg')
        plt.show()
        
    def draw_run_time(titles, train_accs, run_time):
        plt.figure(figsize=(32,16))
        plt.grid()
        plt.subplot(121)
        plt.title('run_time')
        rects = plt.bar(titles, run_time)
        plt.xlabel('Model with different modules')
        plt.ylabel('run_time')
        plt.xticks(titles, titles, rotation=30)
        for rect in rects:  
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(round(height, 2)), size=15, ha='center', va='bottom')

        plt.subplot(122)
        plt.title('epochs')
        train_accs = [str2num_list(train_acc) for train_acc in train_accs]
        epochs = [len(i) for i in train_accs]
        rects = plt.bar(titles, epochs)
        plt.xlabel('Model with different modules')
        plt.ylabel('epochs')
        plt.xticks(titles, titles, rotation=20)
        for rect in rects:  
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(round(height, 2)), size=15, ha='center', va='bottom')
        plt.savefig('runtime_epoches.jpg')
        plt.show()

    result = pd.read_csv(result_file_path, sep=',', names=['title', 'train_accs', 'train_losses', 'dev_accs', 'dev_losses', 'test_loss', 'test_acc', 'run_time'])
    titles = result['title'].to_list()
    train_accs = result['train_accs'].to_list()
    train_losses = result['train_losses'].to_list()
    dev_accs = result['dev_accs'].to_list()
    dev_losses = result['dev_losses'].to_list()
    test_loss = [float(i) for i in result['test_loss'].to_list()]
    test_acc = [float(i) for i in result['test_acc'].to_list()]
    run_time = [float(i) for i in result['run_time'].to_list()]
    
    # draw_acc_loss(titles, train_accs, dev_accs)
    # draw_acc_loss(titles, train_losses, dev_losses)
    # draw_test_matrics(titles, test_acc, test_loss)
    print(train_accs)
    draw_run_time(titles, train_accs, run_time)
    


if __name__ == "__main__":
    graphing('ass_1_0404/module_result.txt')