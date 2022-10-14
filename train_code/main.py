import numpy as np
import os
import shutil

import config
from trainer import MultiSourceTrainer

import evaluate_model

def run_trainer():
    # First, perform source-only training, in order to have source-only models for each domain.
    trainers = {}

    # Clean the experiment folder
    for name in ['weights_path', 'summaries_path']:
        if os.path.exists(os.path.join(config.settings[name], config.settings['exp_name'])):
            shutil.rmtree(os.path.join(config.settings[name], config.settings['exp_name']))

    # Perform source training
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        dom = config.settings['src_datasets'][src_domain_idx]

        print("Training on source domain {}".format(config.settings['src_datasets'][src_domain_idx]))
        
        trainers[dom] = MultiSourceTrainer(src_domain_idx)
        trainers[dom].init_folder_paths()

        while trainers[dom].current_iteration < config.settings['enough_iter'] + 1:
            trainers[dom].train()

            if trainers[dom].current_iteration%trainers[dom].settings['val_after']== 0:
                trainers[dom].check_and_save_weights()

                trainers[dom].set_mode(config.settings['mode']['val'])
                source_acc = trainers[dom].val_over_source_set()
                target_acc = trainers[dom].val_over_target_set()
                print("Obtained target accuracy = {:.4f} on {} and source acc = {:.4f} on {} at iteration {}".format(target_acc, trainers[dom].trgt_domain, \
                    source_acc, trainers[dom].src_domain, trainers[dom].current_iteration))
                
            trainers[dom].current_iteration += 1

    # Load the best source weights
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        dom = config.settings['src_datasets'][src_domain_idx]

        print("Loading data from the source domain {}".format(config.settings['src_datasets'][src_domain_idx]))
        trainers[dom].load_model_weights(it_thresh='enough_iter')

    # Check the accuracy at the start of adaptation is the same as the best one from training
    print()
    for dom in config.settings['src_datasets']:
        target_acc = trainers[dom].val_over_target_set()
        source_acc = trainers[dom].val_over_source_set()
        print("Before adaptation, source acc = {:.4f} and target acc = {:.4f} for src. domain {}".format(source_acc, target_acc, dom))
    print()

    # Learn and save mixing weights
    w_hc, _, _ = evaluate_model.learn_w_high_confidence('enough_iter', .5)
    print("Saving high confidence mixing weights")
    np.savetxt(os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'mixing_weights.txt'), w_hc, fmt='%.5f', newline = ' ')

    # Learn and save pseudo-distribution
    pseudo_target_dist = np.ones(list(config.settings['num_C'].values())[0])
    np.savetxt(os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'pseudo_target_dist.txt'), pseudo_target_dist, fmt='%.5f', newline = ' ')

    # Now, perform adaptation
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        dom = config.settings['src_datasets'][src_domain_idx]
        trainers[dom].mixing_weights =  np.loadtxt(os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'mixing_weights.txt'))

        print("Adapting for source domain {}".format(dom))

        while trainers[dom].current_iteration < config.settings['max_iter'] + 1:
            trainers[dom].train()

            if trainers[dom].current_iteration%trainers[dom].settings['val_after']== 0:
                trainers[dom].check_and_save_weights()

                trainers[dom].set_mode(config.settings['mode']['val'])
                target_acc = trainers[dom].val_over_target_set()
                print("Obtained target accuracy = {:.4f} on {} at iteration {}".format(target_acc, trainers[dom].trgt_domain, trainers[dom].current_iteration))

            if trainers[dom].current_iteration == trainers[dom].settings['max_iter']:
                # make sure the weights are properly saved
                trainers[dom].load_model_weights(it_thresh='max_iter')
                trainers[dom].load_optimizers()
                target_acc_loaded = trainers[dom].val_over_target_set(save_weights=False)
                
                print("Final target accuracy for domain {} = {:.4f}".format(trainers[dom].src_domain, target_acc))
                assert np.abs(target_acc - target_acc_loaded) < 1e-8

                trainers[dom].save_summaries()
                
            trainers[dom].current_iteration += 1

def evaluate():
    # evaluate_model.test_predictions()

    combined_target_perf = []

    eval_tuning_steps = config.settings['eval_tuning_steps']

    # Combine domains based on W2 distance between latent spaces
    w_hat_w2 = evaluate_model.learn_w_w2(num_steps=eval_tuning_steps)
    combined_target_perf.append(evaluate_model.get_target_accuracy(w_hat_w2, 'max_iter')[0])

    # Choosing based on initial presence of high confidence labels
    w_hc, w_hc_raw, data_size = evaluate_model.learn_w_high_confidence('enough_iter', .5)
    eval_data = evaluate_model.get_target_accuracy(w_hc, 'max_iter')
    combined_target_perf.append(eval_data[0])

    print(w_hc, w_hc_raw, data_size, "enough iter")
    for conf in range(10):
        w_hc2, w_hc_raw2, data_size2 = evaluate_model.learn_w_high_confidence('max_iter', conf/10)
        print(w_hc2, w_hc_raw2, data_size2, "max iter", conf/10)

    evaluate_model.get_combined_predictions(w_hc, 'enough_iter')

    evaluate_model.get_combined_predictions(w_hc, 'max_iter')

    # Combine domains evenly
    combined_target_perf.append(evaluate_model.get_target_accuracy(np.ones(len(config.settings['src_datasets'])), 'max_iter')[0])

    individual_src_only_perf, individual_target_perf = evaluate_model.get_individual_performance()

    return combined_target_perf, individual_src_only_perf, individual_target_perf

def main():
    run_trainer()

    print("\n\n\n")

    combined, individual_src_only, individual_target = evaluate()

    print("\n\n\n")

    all_results = np.concatenate([combined, individual_src_only, individual_target])
    all_results = all_results.reshape(1, -1)
    np.savetxt(os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'results.txt'), all_results, fmt='%.5f', newline = ' ')


if __name__ == "__main__":
    main()