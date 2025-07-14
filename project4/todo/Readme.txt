Environment : Kaggle Notebook
Accelerator: GPU T100

1.alexnet-train.ipynb
create model, read train data, training model, read test data and evluation result.
	1st block: Change os.walk('/kaggle/input') to release data path
	3rd block: Change os.chdir('/kaggle/input/nycu-ml-pattern-recognition-hw-4/released/test') to test data path
	10th block: Change torch.save(model.state_dict(), "/kaggle/working/alexnet-weight.pth") to any path which can save model weight
	16th block: Change submission.to_csv('/kaggle/working/submission.csv', index = None) to any path which can save predict result.

2.alexnet-interface.ipynb
create model, read test data, load model weight and evluation result.
	1st block: Change os.walk('/kaggle/input') to release data path
	3rd block: Change os.chdir('/kaggle/input/nycu-ml-pattern-recognition-hw-4/released/test') to test data path
	th block: Change model.load_state_dict(torch.load("/kaggle/input/train/alexnet-weight.pth")) to model weight path
	12th block: Change submission.to_csv('/kaggle/working/submission.csv', index = None) to any path which can save predict result.

3. other_test_code
contain 3 ipynb
alexnet_ablation_test.ipynb is the ablation test with more layer in MIL classifier.
resnet18_train.ipynb use Resnet18 as base model, train model and save model weight.
resnet18_interface.ipynb load Resnet18 weight, evluation training and validataion data, and predict testing data (Need to change path same as alexnet-interface)