# CipherBusters: Automatically Busting Classical Ciphers 👻

John Chung, Alex Ding, Megan Frisella

## Introduction

A cipher is a procedure that defines how to transform plaintext into ciphertext. Cracking a cipher is the task of reverse-engineering a given ciphertext to recover the plaintext without knowledge of the cipher. This is a popular adversarial task used to uncover flaws in encryption schemes with the goal of producing more robust ciphers. Statistical methods have proven powerful for decrypting a variety of simple ciphers using the concept of frequency analysis. Frequency analysis methods are particularly useful for 1:1 ciphers, which are ciphers where there is a 1:1 mapping between characters in the plaintext alphabet and ciphertext alphabet.

Substitution cipher falls under this category. Substitution cipher assigns a random 1:1 mapping of plaintext to ciphertext characters. Transformation from plaintext to ciphertext means replacing each character with the character it is mapped to. Analyzing the frequency of individual characters, pairs of characters (2-grams), or N-grams works well to decrypt this sort of ciphertext because the frequencies can be compared against known frequencies in the plaintext alphabet. Caesar’s cipher is a simplification of substitution cipher, where the 1:1 mapping is defined by shifting the alphabet over N characters. This cipher is considered easier to crack because the mapping between plaintext and ciphertext characters has a consistent offset, so if you figure out the mapping for one character (or N-gram) then you get all of them. Vigènere cipher is slightly more complicated, where multiple different Caesar ciphers are applied at different characters in the plaintext according to some key. 

| Plaintext                                                               | Caesar (key=3)                                                          | Vigènere (key=bust)                                                     | Substitution                                                            |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| the 1935 washington huskies football team was an american football team | wkh 4c68 zdvklqjwrq kxvnlhv irrwedoo whdp zdv dq dphulfdq irrwedoo whdp | u1w kann fbcz1o0b7o 1cbl2wb g86ccu34 uys5 xua to u4xs2uto z67uvs4m dwtn | j1p hial tcf15v0jqv 1bfs5pf uqqj2c66 jpco tcf cv cop85dcv uqqj2c66 jpco |

> Table 1. Examples of all three cipher schemes used in this paper.

You might ask, “How do we know which decryption method to apply if we don’t know which cipher we are trying to crack?” Our work aims to make headway on this problem by utilizing deep learning to create a generalized algorithm that can reliably decrypt multiple different ciphers. 

## Related Work

The idea of taking advantage of the generalizability of deep learning to crack ciphers has been explored in the past. In particular, Focardi et al. trained various simple feed-forward networks using letter frequencies to crack Caesar cipher, Vigènere, and arbitrary substitution ciphers. David Z. expanded on the work to also include affine cipher. Their methodology requires preexisting knowledge about the specific type of cipher, as the model architectures are distinct for different cipher types. Kopal trained a feed-forward network using manually engineered features to classify between different types of ciphers. We build on their work to evaluate whether more powerful sequential models can generalize the decryption pipeline such that a single model can handle cipher identification and decryption of various different simple ciphers while maintaining good performance in these ciphers. 

We aim to develop an even more generalized algorithm by removing the statistical analysis. We hypothesize that a sequential encoding of the ciphertext will mimic this step by learning latent frequency information. Although we don’t expect as high of accuracy as classic statistical methods, a generalized cipher decryption algorithm has much potential for academic interest (i.e. to see how powerful deep learning methods are when applied to increasingly generalized tasks) and practical importance because we may not know which cipher we are trying to crack (e.g. substitution vs. Vigènere) so a generalized decryption algorithm is able to handle multiple cases. We restrict the domain of ciphers we are trying to crack to Caesar, substitution, and Vigènere. We begin with easier tasks (e.g. decrypting one particular Caesar cipher) on simple machine translation algorithms as a proof of concept. As we increase the generality of the task (e.g. decrypting any arbitrary Caesar cipher), we move on to more robust machine translation models. 

## Methodology

Our approach is to identify various decryption tasks of varying difficulty/generality, then develop a machine translation model that is successful on each task. As the tasks increase in generality, we change the architecture to make the model more robust if need be. We identify the following tasks, in order of increasing generality. 

1. Decrypt ciphertext guaranteed to come from one particular Caesar cipher.
2. Decrypt ciphertext that may come from an arbitrary Caesar cipher (“All Caesar” task).
3. Decrypt ciphertext that may come from any arbitrary substitution cipher.
4. Decrypt ciphertext that may come from any arbitrary Vigènere cipher.
5. Decrypt ciphertext that may come from any arbitrary substitution or Vigènere cipher. 

For all tasks, we use a Wikipedia corpus of English text as our body of plaintext. In total, we sample 20,000 Wikipedia articles for a total of 58,222,708 characters. We reduce the character space to 26 English letters, 10 Arabic numbers, and whitespace by lowercasing, normalizing spacing between words, and removing all other characters. We then tokenize and one-hot encode fixed-sized windowed character sequences as matrices. For the train-test split, we use an 80:20 split. 

The plaintext serves as labels for the corresponding ciphertext, which is enciphered differently according to the task. Tasks 1 and 2 use Caesar ciphers, so we enciphered the entire corpus with every Caesar cipher, of which there are 36. Whitespaces characters are ignored during enciphering. For task 1, we randomly choose one Caesar cipher and generate batches of train/test data using the plaintext and the corresponding ciphertext for this cipher. For task 2, we generate batches of train/test data for every cipher then randomly shuffle these together. Thus, each batch comes from one Caesar cipher but the model sees every Caesar cipher during training. For tasks 3, 4, and 5, it is unreasonable to encipher the entire corpus with every substitution and Vigènere cipher because the keyspace is very large (there are 36! keys for substitution cipher and infinitely many for Vigènere). Instead, to generate train/test batches we iterate through batches plaintext and randomly choose a key to encipher each batch. This introduces a training/testing bottleneck because the data is generated on the spot, but it allows the model to see arbitrarily many different substitution and/or Vigènere ciphers during training without having to store ciphertext for arbitrarily many ciphers.

![Figure 1. Our overall pipeline for models [c], [d], and [e]. The encoder and decoder layers can be either LSTMs or transformers. Models [a] and [b] follow the same pipeline, except they do not have the encoder layer.](/images/pipeline.png)
> Figure 1. Our overall pipeline for models [c], [d], and [e]. The encoder and decoder layers can be either LSTMs or transformers. Models [a] and [b] follow the same pipeline, except they do not have the encoder layer.  

In the following, we will describe the progression of model architectures that we built in accordance with increasingly difficult/generalized tasks. We will hint at model performance to motivate the changes to architectures, however we will provide more formal coverage of the results in the next section. We first develop a simple RNN architecture (model \[a\]) and simple transformer architecture (model \[b\]). Each consists of one LSTM or transformer layer and a few fully connected layers. Both quickly learn task 1 but are not as robust to task 2, especially the simple RNN. 

Next, we make the RNN and transformer more robust by adding encoder and decoder layers, which are two LSTMs for the RNN (model \[c\]) and two transformer layers for the transformer (model \[d\]), followed by a few fully connected layers. Both learn task 2 with high accuracy but are not as robust to tasks 3 or 4. To address this, we make the transformer more robust by adding one additional transformer layer each to the encoder and decoder, still followed by multiple fully connected layers (model \[e\]). 

An important hyperparameter for our machine translation task is window size. The basic intuition is that a larger window size may elicit better performance because the model can learn more reliable information about the frequencies of ciphertext characters if more ciphertext is supplied. We ran an experiment with model \[c\] on task 2 where we varied window size and kept all other hyperparameters constant to determine the optimal window size. 

## Results

For all results, the reader can assume that we train models for 20 epochs with a learning rate of 1e-3. Our first task is decrypting one particular Caesar cipher. Our second task is decrypting any arbitrary Caesar cipher, of which there are 36. In Figure 2, we present results from 36 experiments, where each represents the task of decrypting N particular Caesar ciphers. Clearly, N=1 is task 1 and N=36 is task 2. These experiments are trained and tested on the simple RNN for a window size of 50. We observe a negative trend in model performance as the number of Caesar ciphers we train the model to decrypt grows. The simple RNN achieves outstanding performance on task 1 but is not robust to task 2. This motivates us to explore the simple transformer and an encoder-decoder based RNN and transformer to achieve better performance on task 2. 

![Figure 2. Simple RNN model performance significantly declines as we train it on an increasing number of ciphers.](/images/rnn-caesars.png)
> Figure 2. Simple RNN model performance significantly declines as we train it on an increasing number of ciphers. 

Figure 3 presents results for task 2 on the simple RNN, simple transformer, RNN, and transformer. All experiments are trained and tested on a window size of 50. The encoder-decoder architecture of the RNN and transformer, outlined in the pipeline in Figure 1, offers the best performance on this task. For this reason, we move forward with these architectures for tasks 3 and 4. 

| Encrypted                                                                                        | Decrypted                                                                                        |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| mtzwzrt343 nzyqt2x zn4z053p3 3xl24 m54 wtvp 4sp9 34tww zn4z053p3 wp43 yz4 rp4 lsplo zq z523pw6p3 | biologists confirm octopuses smart but like they still octopuses lets not get ahead of ourselves |
| fgy60d ywd7ea9 d0fdwyfe x4ff0d d08wd6e w1f0d 149w77k 20ff492 wyy0bf0z fa xdai9                   | tucker carlson retracts bitter remarks after finally getting accepted to brown                   |
| qjy ymj wzqnsl hqfxxjx ywjrgqj fy f htrrzsnxynh wj0tqzynts                                       | let the ruling classes tremble at a communistic revolution                                       |

> Table 2. Examples of trained transformer decrypting arbitrary Caesar ciphers. 

![Figure 3. The encoder-decoder architectures exhibit the best performance on the “All Caesar” task.](/images/all-caesars.png)
> Figure 3. The encoder-decoder architectures exhibit the best performance on the “All Caesar” task. 

Before presenting results for tasks 3 and 4, we motivate our chosen window size of 50 by the following experiment. In Figure 4, we present results from 13 window size experiments, where each experiment is the test accuracy of the model [c] on the “All Caesar” task. Each experiment has a different window size. We observe that the optimal window size is ~64. Window sizes larger than this maximum begin to see decreases in performance, which we did not expect. This contradicts our intuition that the model learns more reliable information about the frequencies of ciphertext characters if more ciphertext is supplied. 

![Figure 4. The optimal window size for the RNN model on the 36 Caesar cipher task is ~64. Window size does not always positively vary with accuracy.](/images/rnn-windows.png)
> Figure 4. The optimal window size for the RNN model on the 36 Caesar cipher task is ~64. Window size does not always positively vary with accuracy.

Finally, we present results for tasks 3 and 4. Figure 5 displays performance for the transformer (model [d]) and robust transformer (model [e]) on task 3 (substitution) and task 4 (Vigènere). Worse performance on substitution compared to Vigènere indicates that the substitution task is more difficult than the Vigènere task. This is expected because Vigènere is a complexification of Caesar’s cipher, for which the models have shown extremely high performance. Overall, the robust transformer does not offer major performance gain compared to the transformer. Although 56% accuracy on Vigènere has merit given the infinite keyspace of Vigènere ciphers, it is not nearly high enough to produce sensible plaintext. Thus, we believe that these methods have reached their limit with regard to generalizing simple decryption tasks. 

![Figure 5. The transformer and robust transformer exhibit better performance on the Vigènere task compared to the substitution task. The robust transformer only offers slight performance gains.](/images/vigenere-substitution.png)
> Figure 5. The transformer and robust transformer exhibit better performance on the Vigènere task compared to the substitution task. The robust transformer only offers slight performance gains.

## Challenges
Our challenges pertained to our model architecture and training methods. Our first major challenge regarding model architecture was that teacher forcing failed during inference time. Initially, we built encoder-decoder RNN and transformer models that used teacher forcing during training (i.e. the model saw the plaintext during training). When testing these models without teacher forcing during inference time, they did not generalize to new ciphertext. To overcome this challenge, we started from scratch on with the simple RNN and simple transformer architectures, which return probability distributions over one-hot encodings for input ciphertext without ever seeing the plaintext. This method was significantly more successful. We eventually returned to an encoder-decoder architecture without teacher forcing. Additionally, we had to determine the best window size for our RNN model that struck a balance between accuracy and memory. Finally, our models did not perform as well on the generalized tasks.

As for our training methods, we could not encode the training plaintext for the substitution and Vigènere tasks in the way we did with the Caesar task because the key space for these ciphers is enormous. Instead, we dynamically generated training data for each batch by randomly choosing a cipher and encoding a batch of plaintext. Although this method introduced a training bottleneck, the model was able to see arbitrarily many different ciphers during training without significant storage needs.

## Conclusion
We accomplished Tasks 1 and 2 outlined in our Methodology. Our best models achieve near-perfect accuracy for the Caesar cipher task. However, our models underperformed for Tasks 3 and 4, exhibiting only 38% accuracy for the substitution cipher task and 56% accuracy for the Vigènere cipher task. Unfortunately, we were unable to pursue Task 5 due to the limited timeline.

The latter half of our tasks were unsuccessful because we incorrectly hypothesized that a larger window size leads to better performance. Figure 4 suggests that the model is not learning latent information about frequencies. Perhaps while the encoder guesses the key and the decoder iterates through each character of the ciphertext, the LSTM slowly loses information about the encoded key. Additionally, our model exhibited better performance for the Vigènere cipher task than the substitution cipher task because of differences between the ciphers’ complexities. The Vigènere cipher, an extension of the Caesar cipher, shifts each character by some arbitrary amount. However, the substitution cipher does not make these shifts, which makes it more complicated and difficult to decrypt.

One direction of future work is to add an adversarial component to our training pipeline to punish implausible decryptions. As it is, our model often devolves into guessing the most common letters during training. The adversarial loss can help mitigate such loss-gaming behavior. Another direction of work is to handcraft features about ciphertext, such as frequency analysis, and feed that as part of the input to the model. This may lead to better performance by limiting the size of the problem and encouraging more probable outcomes. Coupling the model with an adversarial component and frequency analysis may address the underperformance seen in the generalization tasks.

## Bibliography
1. Focardi, R. and Luccio, F.L., 2018. Neural Cryptanalysis of Classical Ciphers. In ICTCS (pp. 104-115).
2. Kopal, N., 2020, May. Of Ciphers and Neurons–Detecting the Type of Ciphers Using Artificial Neural Networks. In Proceedings of the 3rd International Conference on Historical Cryptology HistoCrypt 2020 (No. 171, pp. 77-86). Linköping University Electronic Press.
3. Z., D., 2021, February. An Exposition of Neural Cryptanalysis of Classical Ciphers. 
