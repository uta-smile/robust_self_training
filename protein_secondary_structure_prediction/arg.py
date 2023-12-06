import argparse


def getArgparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', metavar='device', type=str, default='cpu',
                        help='Device to run on, Either: cpu or coda (default; cpu)')

    parser.add_argument('--teacher_fpath', metavar='teacher_fpath', type=str, default='model/teacher/0_best.pkl',
                        help='this is for save model path (default; default)')

    parser.add_argument('--learn_rate', metavar='learn_rate', type=float, default=0.0001,
                        help='this is learn_rate (default; 0.0001)')

    parser.add_argument('--save_path', metavar='save_path', type=str, default='model/',
                        help='this is for save model path (default; default)')

    parser.add_argument('--options', metavar='options', type=str, default='test',
                        help='this are three options for train: pretrain/fine_tuning/test/train_labelled.txt (default; '
                             'test)')

    parser.add_argument('--dropout', metavar='dropout', type=float, default=0.0,
                        help='this is dropout (default; 0.0)')

    parser.add_argument('--num_workers', metavar='num_workers', type=int, default=12,
                        help='this is num_workers (default; 12)')

    parser.add_argument('--data_path', metavar='data_path', type=str, default='dataset/predicted',
                        help='this is for save data path (default; dataset/predicted)')

    parser.add_argument('--iteration', metavar='iteration', type=int, default=3,
                        help='this is iteration (default; 3)')

    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=64,
                        help='This is batch_size (default; 64)',
                        )

    parser.add_argument('--epochs', metavar='epochs', type=int, default=10,
                        help='this is epochs (default; 10)')

    parser.add_argument('--loss_name', metavar='loss_name', type=str, default='CE',
                        help='this is loss_name (default; CE)')

    parser.add_argument('--loss_weights', metavar='loss_weights', type=str, default='0.7,0.3',
                        help='this is loss_name (default; 10)')

    parser.add_argument('--is_init', metavar='is_init', type=bool, default=False,
                        help='this is loss_name (default; 10)')

    parser.add_argument('--load_model', metavar='load_model', type=bool, default=False,
                        help='this is load_con (default; False)')

    parser.add_argument('--load_student', metavar='load_student', type=str, default='last',
                        help='this is load_student (default; last)')

    parser.add_argument('--test_model_name', metavar='test_model_name', type=str, default='best',
                        help='this is test_model_name (default; 10)')

    parser.add_argument('--gce_q', metavar='gce_q', type=float, default=0.7,
                        help='this is GCE q (default; 0.7)')

    parser.add_argument('--dmice_p', metavar='dmice_p', type=float, default=0.1,
                        help='this is dmice_p  (default; 0.1)')

    parser.add_argument('--is_homologous', metavar='--is_homologous', type=bool, default=False,
                        help='this is --is_homologous (default; False)')

    parser.add_argument('--model', metavar='model', type=str, default='S4PRED',
                        help='which model (S4PRED/Ensemble)')

    args = parser.parse_args()
    return vars(args)

# CUDA_VISIBLE_DEVICES=1,2  python main.py --device cuda --options train_labelled.txt --epochs 10 --batch_size 64
# --save_path model/train_labelled_04/

# CUDA_VISIBLE_DEVICES=3 python selftrain.py --device cuda  --iteration 3 --dropout 0.5 --epochs 5 --save_path model/selftrain_labelled02/ --data_path dataset/predicted05


