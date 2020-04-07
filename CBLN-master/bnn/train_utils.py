import tensorflow as tf
import numpy as np

def reduce_entropy(X, axis=-1):
    """
    calculate the entropy over axis and reduce that axis
    :param X:
    :param axis:
    :return:
    """
    return -1 * np.sum(X * np.log(X+1E-12), axis=axis)

def calc_risk(preds, labels=None):
    """
    Calculates the parameters we can possibly use to examine risk of a neural net
    :param preds: preds in shape [num_runs, num_batch, num_classes]
    :param labels:
    :return:
    """
    #if isinstance(preds, list):
    #    preds = np.stack(preds)
    # preds in shape [num_runs, num_batch, num_classes]
    num_runs, num_batch = preds.shape[:2]

    ave_preds = np.mean(preds, axis=0)
    pred_class = np.argmax(ave_preds, axis=1)

    # entropy of the posterior predictive
    entropy = reduce_entropy(ave_preds, axis=1)

    # Expected entropy of the predictive under the parameter posterior
    entropy_exp = np.mean(reduce_entropy(preds, axis=2), axis=0)
    mutual_info = entropy - entropy_exp  # Equation 2 of https://arxiv.org/pdf/1711.08244.pdf

    # Average and variance of softmax for the predicted class
    variance = np.std(preds[:, range(num_batch), pred_class], 0)
    ave_softmax = np.mean(preds[:, range(num_batch), pred_class], 0)

    # And calculate accuracy if we know the labels
    if labels is not None:
        correct = np.equal(pred_class, labels)
    else:
        correct = None
    predictive_score = ave_preds[np.arange(num_batch),pred_class]
    return [entropy, mutual_info, variance, ave_softmax, correct,predictive_score]

def em_predict(predictions,y):
    avg_pred = np.mean(predictions,axis=0)
    risk = calc_risk(predictions)
    acc = np.mean(np.equal(np.argmax(y,1), np.argmax(avg_pred, axis=-1)))
    return acc,risk,risk[-1]

def eval(net,sess,num_task,writer,test_init,test_accs,params_idx=None,disp=True,record=True):
    avg_acc_all = 0.0
    current_acc = 0.0
    for test_idx in range(num_task):
        sess.run(test_init[test_idx])
        if params_idx is not None:
            #improving
            net.set_task_params(sess,params_idx[test_idx])
        avg_acc = 0.0
        #for _ in range(1000):
        num_test = 0
        while True:
            try:
                if writer is not None:
                    acc,summaries,step = sess.run([net.accuracy,net.summary_op,net.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
                else:
                    acc,step = sess.run([net.accuracy,net.gstep])
                num_test += 1
                avg_acc += acc
            except tf.errors.OutOfRangeError:
                break

        
        if writer is not None:
            writer.add_summary(summaries,global_step = step)
        if record:
            test_accs[test_idx].append(avg_acc / num_test)
        avg_acc_all += avg_acc / num_test

    if record:
        test_accs['avg'].append(avg_acc_all / num_task)

    return avg_acc_all / num_task


def em_eval(net,sess,num_task,writer,testsets,test_accs,disp=True,record=True,num_runs=200,search_best=True):
    def make_prediction(data,label):
        predictions = []
        total_acc = 0.0
        for _ in range(num_runs):
            pred, em_acc = sess.run([net.predictions,net.em_accuracy],feed_dict={net.x_placeholder:data,net.y_placeholder:label})
            predictions.append(pred)
            total_acc += em_acc
        return np.array(predictions),total_acc/num_runs

    avg_acc_all = 0.0
    params_idx_list = []
    total_iter = num_task * len(net.params_mean.keys())
    iter_step = 0
    correct = True

    for test_idx in range(num_task):
        avg_acc = []
        avg_uncertainty = []
        
        
        for params_idx in net.params_mean.keys():
            print('Getting Idex {}/{} ...'.format(iter_step,total_iter),end='\r')
            iter_step += 1
            net.set_task_params(sess,params_idx)
            avg_acc.append(0.0)
            avg_uncertainty.append(0.0)
            
            

            for iters in range(1):
                pred_idx = np.random.choice(np.arange(testsets[test_idx][0].shape[0]),200)
                test_data = testsets[test_idx][0][pred_idx]
                test_label = testsets[test_idx][1][pred_idx]
            #while True:
                try:
                    
                    predictions,acc = make_prediction(test_data,test_label)
                    step = sess.run(net.gstep)
                    acc,uncertainty,scores = em_predict(predictions,test_label)
                except tf.errors.OutOfRangeError:
                    pass

                avg_uncertainty[params_idx] += uncertainty[2]
                
                


        info = 'Task {} :  {} th set of parameters has minimal uncertainty : Correct !'.format(test_idx,np.argmin(np.mean(avg_uncertainty,axis=1)))
        params_idx_list.append(np.argmin(np.mean(avg_uncertainty,axis=1)))
        min_idx = np.argmin(np.mean(avg_uncertainty,axis=1))
        if search_best and min_idx != test_idx and num_task<10:
        	correct = False
        	info = 'Task {} :  {} th set of parameters has minimal uncertainty : Wrong !'.format(test_idx,np.argmin(np.mean(avg_uncertainty,axis=1)))
        	print(info)
        	print('The model cannot identify the test data correctly ...')
        	print('Search for a new model, {} th running start ...'.format(net.num_runs+1))
        	net.num_runs += 1
        	break
        print(info)

    return params_idx_list,correct

def em_train(model,sess,num_epoch,trainset,testsets,train_init,test_init,lams=[0.01],search_best=True):
    
    model.initialize_default_params(sess)
    dp = False
    test_accs = {}
    test_accs['avg'] = []
    for t in range(len(testsets)):
        test_accs[t] = []
    print('Training start ...')

    #writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
    writer = None
    num_task = len(trainset)
    sess.run(model.lams.assign(lams[0]))


    for idx in range(num_task):
        model.reset(sess)

        print('Training {} th Task  ...'.format(idx+1))
        #  Training Start
        for e in range(num_epoch):
            sess.run(train_init[idx])
            try:
                while True:
                    _,step = sess.run([model.train_op,model.gstep])
                    
            except tf.errors.OutOfRangeError:
            	pass
                

        model.store_params(idx)

              
    print('Merging Process Start ... ')
    model.st_smooth(n_component=num_task,dp=dp,thresh_hold=0.5/num_task)
  
    print('Evaluating the Uncertainty ... ')
    param_idx,correct = em_eval(model,sess,num_task,None,testsets,test_accs,record=False,disp=False,search_best=search_best)
    if correct:
    	acc = eval(model,sess,num_task,None,test_init,test_accs,params_idx=param_idx,record=True,disp=False)
    	print('Find the best model after searching  {} times, Final Average Accuracy for all the Tasks : {}'.format(model.num_runs,acc))
    else:
    	em_train(model,sess,num_epoch,trainset,testsets,train_init,test_init,lams=[0],search_best=search_best)

   
    