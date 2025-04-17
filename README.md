# Latest Adversarial Attack Papers
**update at 2025-04-17 11:39:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Human Aligned Compression for Robust Models**

cs.CV

Presented at the Workshop AdvML at CVPR 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12255v1) [paper-pdf](http://arxiv.org/pdf/2504.12255v1)

**Authors**: Samuel Räber, Andreas Plesner, Till Aczel, Roger Wattenhofer

**Abstract**: Adversarial attacks on image models threaten system robustness by introducing imperceptible perturbations that cause incorrect predictions. We investigate human-aligned learned lossy compression as a defense mechanism, comparing two learned models (HiFiC and ELIC) against traditional JPEG across various quality levels. Our experiments on ImageNet subsets demonstrate that learned compression methods outperform JPEG, particularly for Vision Transformer architectures, by preserving semantically meaningful content while removing adversarial noise. Even in white-box settings where attackers can access the defense, these methods maintain substantial effectiveness. We also show that sequential compression--applying rounds of compression/decompression--significantly enhances defense efficacy while maintaining classification performance. Our findings reveal that human-aligned compression provides an effective, computationally efficient defense that protects the image features most relevant to human and machine understanding. It offers a practical approach to improving model robustness against adversarial threats.



## **2. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.



## **3. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.



## **4. Formal Verification of Graph Convolutional Networks with Uncertain Node Features and Uncertain Graph Structure**

cs.LG

published at Transactions on Machine Learning Research (TMLR) 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2404.15065v2) [paper-pdf](http://arxiv.org/pdf/2404.15065v2)

**Authors**: Tobias Ladner, Michael Eichelbeck, Matthias Althoff

**Abstract**: Graph neural networks are becoming increasingly popular in the field of machine learning due to their unique ability to process data structured in graphs. They have also been applied in safety-critical environments where perturbations inherently occur. However, these perturbations require us to formally verify neural networks before their deployment in safety-critical environments as neural networks are prone to adversarial attacks. While there exists research on the formal verification of neural networks, there is no work verifying the robustness of generic graph convolutional network architectures with uncertainty in the node features and in the graph structure over multiple message-passing steps. This work addresses this research gap by explicitly preserving the non-convex dependencies of all elements in the underlying computations through reachability analysis with (matrix) polynomial zonotopes. We demonstrate our approach on three popular benchmark datasets.



## **5. Attribute Inference Attacks for Federated Regression Tasks**

cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2411.12697v2) [paper-pdf](http://arxiv.org/pdf/2411.12697v2)

**Authors**: Francesco Diana, Othmane Marfoq, Chuan Xu, Giovanni Neglia, Frédéric Giroire, Eoin Thomas

**Abstract**: Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks.



## **6. RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning**

cs.CR

16 pages, 10 Figures

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2502.08989v2) [paper-pdf](http://arxiv.org/pdf/2502.08989v2)

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Dineshkumar Singh, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security.



## **7. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2503.12793v3) [paper-pdf](http://arxiv.org/pdf/2503.12793v3)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.



## **8. SemDiff: Generating Natural Unrestricted Adversarial Examples via Semantic Attributes Optimization in Diffusion Models**

cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11923v1) [paper-pdf](http://arxiv.org/pdf/2504.11923v1)

**Authors**: Zeyu Dai, Shengcai Liu, Rui He, Jiahao Wu, Ning Lu, Wenqi Fan, Qing Li, Ke Tang

**Abstract**: Unrestricted adversarial examples (UAEs), allow the attacker to create non-constrained adversarial examples without given clean samples, posing a severe threat to the safety of deep learning models. Recent works utilize diffusion models to generate UAEs. However, these UAEs often lack naturalness and imperceptibility due to simply optimizing in intermediate latent noises. In light of this, we propose SemDiff, a novel unrestricted adversarial attack that explores the semantic latent space of diffusion models for meaningful attributes, and devises a multi-attributes optimization approach to ensure attack success while maintaining the naturalness and imperceptibility of generated UAEs. We perform extensive experiments on four tasks on three high-resolution datasets, including CelebA-HQ, AFHQ and ImageNet. The results demonstrate that SemDiff outperforms state-of-the-art methods in terms of attack success rate and imperceptibility. The generated UAEs are natural and exhibit semantically meaningful changes, in accord with the attributes' weights. In addition, SemDiff is found capable of evading different defenses, which further validates its effectiveness and threatening.



## **9. RAB$^2$-DEF: Dynamic and explainable defense against adversarial attacks in Federated Learning to fair poor clients**

cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.08244v2) [paper-pdf](http://arxiv.org/pdf/2410.08244v2)

**Authors**: Nuria Rodríguez-Barroso, M. Victoria Luzón, Francisco Herrera

**Abstract**: At the same time that artificial intelligence is becoming popular, concern and the need for regulation is growing, including among other requirements the data privacy. In this context, Federated Learning is proposed as a solution to data privacy concerns derived from different source data scenarios due to its distributed learning. The defense mechanisms proposed in literature are just focused on defending against adversarial attacks and the performance, leaving aside other important qualities such as explainability, fairness to poor quality clients, dynamism in terms of attacks configuration and generality in terms of being resilient against different kinds of attacks. In this work, we propose RAB$^2$-DEF, a $\textbf{r}$esilient $\textbf{a}$gainst $\textbf{b}\text{yzantine}$ and $\textbf{b}$ackdoor attacks which is $\textbf{d}$ynamic, $\textbf{e}$xplainable and $\textbf{f}$air to poor clients using local linear explanations. We test the performance of RAB$^2$-DEF in image datasets and both byzantine and backdoor attacks considering the state-of-the-art defenses and achieve that RAB$^2$-DEF is a proper defense at the same time that it boosts the other qualities towards trustworthy artificial intelligence.



## **10. Support is All You Need for Certified VAE Training**

cs.LG

21 pages, 3 figures, ICLR '25

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11831v1) [paper-pdf](http://arxiv.org/pdf/2504.11831v1)

**Authors**: Changming Xu, Debangshu Banerjee, Deepak Vasisht, Gagandeep Singh

**Abstract**: Variational Autoencoders (VAEs) have become increasingly popular and deployed in safety-critical applications. In such applications, we want to give certified probabilistic guarantees on performance under adversarial attacks. We propose a novel method, CIVET, for certified training of VAEs. CIVET depends on the key insight that we can bound worst-case VAE error by bounding the error on carefully chosen support sets at the latent layer. We show this point mathematically and present a novel training algorithm utilizing this insight. We show in an extensive evaluation across different datasets (in both the wireless and vision application areas), architectures, and perturbation magnitudes that our method outperforms SOTA methods achieving good standard performance with strong robustness guarantees.



## **11. PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection**

cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2406.05826v2) [paper-pdf](http://arxiv.org/pdf/2406.05826v2)

**Authors**: Wei Li, Pin-Yu Chen, Sijia Liu, Ren Wang

**Abstract**: Deep neural networks are susceptible to backdoor attacks, where adversaries manipulate model predictions by inserting malicious samples into the training data. Currently, there is still a significant challenge in identifying suspicious training data to unveil potential backdoor samples. In this paper, we propose a novel method, Prediction Shift Backdoor Detection (PSBD), leveraging an uncertainty-based approach requiring minimal unlabeled clean validation data. PSBD is motivated by an intriguing Prediction Shift (PS) phenomenon, where poisoned models' predictions on clean data often shift away from true labels towards certain other labels with dropout applied during inference, while backdoor samples exhibit less PS. We hypothesize PS results from the neuron bias effect, making neurons favor features of certain classes. PSBD identifies backdoor training samples by computing the Prediction Shift Uncertainty (PSU), the variance in probability values when dropout layers are toggled on and off during model inference. Extensive experiments have been conducted to verify the effectiveness and efficiency of PSBD, which achieves state-of-the-art results among mainstream detection methods. The code is available at https://github.com/WL-619/PSBD.



## **12. Towards Safe Synthetic Image Generation On the Web: A Multimodal Robust NSFW Defense and Million Scale Dataset**

cs.CV

Short Paper The Web Conference

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11707v1) [paper-pdf](http://arxiv.org/pdf/2504.11707v1)

**Authors**: Muhammad Shahid Muneer, Simon S. Woo

**Abstract**: In the past years, we have witnessed the remarkable success of Text-to-Image (T2I) models and their widespread use on the web. Extensive research in making T2I models produce hyper-realistic images has led to new concerns, such as generating Not-Safe-For-Work (NSFW) web content and polluting the web society. To help prevent misuse of T2I models and create a safer web environment for users features like NSFW filters and post-hoc security checks are used in these models. However, recent work unveiled how these methods can easily fail to prevent misuse. In particular, adversarial attacks on text and image modalities can easily outplay defensive measures. %Exploiting such leads to the growing concern of preventing adversarial attacks on text and image modalities. Moreover, there is currently no robust multimodal NSFW dataset that includes both prompt and image pairs and adversarial examples. This work proposes a million-scale prompt and image dataset generated using open-source diffusion models. Second, we develop a multimodal defense to distinguish safe and NSFW text and images, which is robust against adversarial attacks and directly alleviates current challenges. Our extensive experiments show that our model performs well against existing SOTA NSFW detection methods in terms of accuracy and recall, drastically reducing the Attack Success Rate (ASR) in multimodal adversarial attack scenarios. Code: https://github.com/shahidmuneer/multimodal-nsfw-defense.



## **13. Learning to Learn Transferable Generative Attack for Person Re-Identification**

cs.CV

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2409.04208v2) [paper-pdf](http://arxiv.org/pdf/2409.04208v2)

**Authors**: Yuan Bian, Min Liu, Xueping Wang, Yunfeng Ma, Yaonan Wang

**Abstract**: Deep learning-based person re-identification (re-id) models are widely employed in surveillance systems and inevitably inherit the vulnerability of deep networks to adversarial attacks. Existing attacks merely consider cross-dataset and cross-model transferability, ignoring the cross-test capability to perturb models trained in different domains. To powerfully examine the robustness of real-world re-id models, the Meta Transferable Generative Attack (MTGA) method is proposed, which adopts meta-learning optimization to promote the generative attacker producing highly transferable adversarial examples by learning comprehensively simulated transfer-based cross-model\&dataset\&test black-box meta attack tasks. Specifically, cross-model\&dataset black-box attack tasks are first mimicked by selecting different re-id models and datasets for meta-train and meta-test attack processes. As different models may focus on different feature regions, the Perturbation Random Erasing module is further devised to prevent the attacker from learning to only corrupt model-specific features. To boost the attacker learning to possess cross-test transferability, the Normalization Mix strategy is introduced to imitate diverse feature embedding spaces by mixing multi-domain statistics of target models. Extensive experiments show the superiority of MTGA, especially in cross-model\&dataset and cross-model\&dataset\&test attacks, our MTGA outperforms the SOTA methods by 21.5\% and 11.3\% on mean mAP drop rate, respectively. The code of MTGA will be released after the paper is accepted.



## **14. Chypnosis: Stealthy Secret Extraction using Undervolting-based Static Side-channel Attacks**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11633v1) [paper-pdf](http://arxiv.org/pdf/2504.11633v1)

**Authors**: Kyle Mitard, Saleh Khalaj Monfared, Fatemeh Khojasteh Dana, Shahin Tajik

**Abstract**: There is a growing class of static physical side-channel attacks that allow adversaries to extract secrets by probing the persistent state of a circuit. Techniques such as laser logic state imaging (LLSI), impedance analysis (IA), and static power analysis fall into this category. These attacks require that the targeted data remain constant for a specific duration, which often necessitates halting the circuit's clock. Some methods additionally rely on modulating the chip's supply voltage to probe the circuit. However, tampering with the clock or voltage is typically assumed to be detectable, as secure chips often deploy sensors that erase sensitive data upon detecting such anomalies. Furthermore, many secure devices use internal clock sources, making external clock control infeasible. In this work, we introduce a novel class of static side-channel attacks, called Chypnosis, that enables adversaries to freeze a chip's internal clock by inducing a hibernation state via rapid undervolting, and then extracting secrets using static side-channels. We demonstrate that, by rapidly dropping a chip's voltage below the standard nominal levels, the attacker can bypass the clock and voltage sensors and put the chip in a so-called brownout condition, in which the chip's transistors stop switching, but volatile memories (e.g., Flip-flops and SRAMs) still retain their data. We test our attack on AMD FPGAs by putting them into hibernation. We show that not only are all clock sources deactivated, but various clock and voltage sensors also fail to detect the tamper event. Afterward, we present the successful recovery of secret bits from a hibernated chip using two static attacks, namely, LLSI and IA. Finally, we discuss potential countermeasures which could be integrated into future designs.



## **15. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **16. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.



## **17. Slice+Slice Baby: Generating Last-Level Cache Eviction Sets in the Blink of an Eye**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11208v1) [paper-pdf](http://arxiv.org/pdf/2504.11208v1)

**Authors**: Bradley Morgan, Gal Horowitz, Sioli O'Connell, Stephan van Schaik, Chitchanok Chuengsatiansup, Daniel Genkin, Olaf Maennel, Paul Montague, Eyal Ronen, Yuval Yarom

**Abstract**: An essential step for mounting cache attacks is finding eviction sets, collections of memory locations that contend on cache space. On Intel processors, one of the main challenges for identifying contending addresses is the sliced cache design, where the processor hashes the physical address to determine where in the cache a memory location is stored. While past works have demonstrated that the hash function can be reversed, they also showed that it depends on physical address bits that the adversary does not know.   In this work, we make three main contributions to the art of finding eviction sets. We first exploit microarchitectural races to compare memory access times and identify the cache slice to which an address maps. We then use the known hash function to both reduce the error rate in our slice identification method and to reduce the work by extrapolating slice mappings to untested memory addresses. Finally, we show how to propagate information on eviction sets across different page offsets for the hitherto unexplored case of non-linear hash functions.   Our contributions allow for entire LLC eviction set generation in 0.7 seconds on the Intel i7-9850H and 1.6 seconds on the i9-10900K, both using non-linear functions. This represents a significant improvement compared to state-of-the-art techniques taking 9x and 10x longer, respectively.



## **18. R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning**

cs.LG

CVPR 2025

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11195v1) [paper-pdf](http://arxiv.org/pdf/2504.11195v1)

**Authors**: Lijun Sheng, Jian Liang, Zilei Wang, Ran He

**Abstract**: Vision-language models (VLMs), such as CLIP, have gained significant popularity as foundation models, with numerous fine-tuning methods developed to enhance performance on downstream tasks. However, due to their inherent vulnerability and the common practice of selecting from a limited set of open-source models, VLMs suffer from a higher risk of adversarial attacks than traditional vision models. Existing defense techniques typically rely on adversarial fine-tuning during training, which requires labeled data and lacks of flexibility for downstream tasks. To address these limitations, we propose robust test-time prompt tuning (R-TPT), which mitigates the impact of adversarial attacks during the inference stage. We first reformulate the classic marginal entropy objective by eliminating the term that introduces conflicts under adversarial conditions, retaining only the pointwise entropy minimization. Furthermore, we introduce a plug-and-play reliability-based weighted ensembling strategy, which aggregates useful information from reliable augmented views to strengthen the defense. R-TPT enhances defense against adversarial attacks without requiring labeled training data while offering high flexibility for inference tasks. Extensive experiments on widely used benchmarks with various attacks demonstrate the effectiveness of R-TPT. The code is available in https://github.com/TomSheng21/R-TPT.



## **19. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.



## **20. Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.10329v2) [paper-pdf](http://arxiv.org/pdf/2411.10329v2)

**Authors**: Huming Qiu, Guanxu Chen, Mi Zhang, Xiaohan Zhang, Xiaoyu You, Min Yang

**Abstract**: In recent years, text-to-image (T2I) generation models have made significant progress in generating high-quality images that align with text descriptions. However, these models also face the risk of unsafe generation, potentially producing harmful content that violates usage policies, such as explicit material. Existing safe generation methods typically focus on suppressing inappropriate content by erasing undesired concepts from visual representations, while neglecting to sanitize the textual representation. Although these methods help mitigate the risk of misuse to some extent, their robustness remains insufficient when dealing with adversarial attacks.   Given that semantic consistency between input text and output image is a core requirement of T2I models, we identify that textual representations are likely the primary source of unsafe generation. To this end, we propose Embedding Sanitizer (ES), which enhances the safety of T2I models by sanitizing inappropriate concepts in prompt embeddings. To our knowledge, ES is the first interpretable safe generation framework that assigns a score to each token in the prompt to indicate its potential harmfulness. In addition, ES adopts a plug-and-play modular design, offering compatibility for seamless integration with various T2I models and other safeguards. Evaluations on five prompt benchmarks show that ES outperforms eleven existing safeguard baselines, achieving state-of-the-art robustness while maintaining high-quality image generation.



## **21. Helper-Friendly Latency-Bounded Mitigation Strategies against Reactive Jamming Adversaries**

cs.IT

16 pages

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11110v1) [paper-pdf](http://arxiv.org/pdf/2504.11110v1)

**Authors**: Soumita Hazra, J. Harshan

**Abstract**: Due to the recent developments in the field of full-duplex radios and cognitive radios, a new class of reactive jamming attacks has gained attention wherein an adversary transmits jamming energy over the victim's frequency band and also monitors various energy statistics in the network so as to detect countermeasures, thereby trapping the victim. Although cooperative mitigation strategies against such security threats exist, they are known to incur spectral-efficiency loss on the helper node, and are also not robust to variable latency-constraints on victim's messages. Identifying these research gaps in existing countermeasures against reactive jamming attacks, we propose a family of helper-friendly cooperative mitigation strategies that are applicable for a wide-range of latency-requirements on the victim's messages as well as practical radio hardware at the helper nodes. The proposed strategies are designed to facilitate reliable communication for the victim, without compromising the helper's spectral efficiency and also minimally disturbing the various energy statistics in the network. For theoretical guarantees on their efficacy, interesting optimization problems are formulated on the choice of the underlying parameters, followed by extensive mathematical analyses on their error-performance and covertness. Experimental results indicate that the proposed strategies should be preferred over the state-of-the-art methods when the helper node is unwilling to compromise on its error performance for assisting the victim.



## **22. Token-Level Constraint Boundary Search for Jailbreaking Text-to-Image Models**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11106v1) [paper-pdf](http://arxiv.org/pdf/2504.11106v1)

**Authors**: Jiangtao Liu, Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Recent advancements in Text-to-Image (T2I) generation have significantly enhanced the realism and creativity of generated images. However, such powerful generative capabilities pose risks related to the production of inappropriate or harmful content. Existing defense mechanisms, including prompt checkers and post-hoc image checkers, are vulnerable to sophisticated adversarial attacks. In this work, we propose TCBS-Attack, a novel query-based black-box jailbreak attack that searches for tokens located near the decision boundaries defined by text and image checkers. By iteratively optimizing tokens near these boundaries, TCBS-Attack generates semantically coherent adversarial prompts capable of bypassing multiple defensive layers in T2I models. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art jailbreak attacks across various T2I models, including securely trained open-source models and commercial online services like DALL-E 3. TCBS-Attack achieves an ASR-4 of 45\% and an ASR-1 of 21\% on jailbreaking full-chain T2I models, significantly surpassing baseline methods.



## **23. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks Against GNN-Based Fraud Detectors**

cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.18370v3) [paper-pdf](http://arxiv.org/pdf/2412.18370v3)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.



## **24. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2312.04960v4) [paper-pdf](http://arxiv.org/pdf/2312.04960v4)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) have emerged as a fundamental architecture and serve as the backbone of modern vision-language models. Despite their impressive performance, ViTs exhibit notable vulnerability to evasion attacks, necessitating the development of specialized Adversarial Training (AT) strategies tailored to their unique architecture. While a direct solution might involve applying existing AT methods to ViTs, our analysis reveals significant incompatibilities, particularly with state-of-the-art (SOTA) approaches such as Generalist (CVPR 2023) and DBAT (USENIX Security 2024). This paper presents a systematic investigation of adversarial robustness in ViTs and provides a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained via derived MI bounds. Building on this insight, we propose a self-supervised AT method, MIMIR, that employs an MI penalty to facilitate adversarial pre-training by masked image modeling with autoencoders. Extensive experiments on CIFAR-10, Tiny-ImageNet, and ImageNet-1K show that MIMIR can consistently provide improved natural and robust accuracy, where MIMIR outperforms SOTA AT results on ImageNet-1K. Notably, MIMIR demonstrates superior robustness against unforeseen attacks and common corruption data and can also withstand adaptive attacks where the adversary possesses full knowledge of the defense mechanism.



## **25. RAID: An In-Training Defense against Attribute Inference Attacks in Recommender Systems**

cs.IR

17 pages

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11510v1) [paper-pdf](http://arxiv.org/pdf/2504.11510v1)

**Authors**: Xiaohua Feng, Yuyuan Li, Fengyuan Yu, Ke Xiong, Junjie Fang, Li Zhang, Tianyu Du, Chaochao Chen

**Abstract**: In various networks and mobile applications, users are highly susceptible to attribute inference attacks, with particularly prevalent occurrences in recommender systems. Attackers exploit partially exposed user profiles in recommendation models, such as user embeddings, to infer private attributes of target users, such as gender and political views. The goal of defenders is to mitigate the effectiveness of these attacks while maintaining recommendation performance. Most existing defense methods, such as differential privacy and attribute unlearning, focus on post-training settings, which limits their capability of utilizing training data to preserve recommendation performance. Although adversarial training extends defenses to in-training settings, it often struggles with convergence due to unstable training processes. In this paper, we propose RAID, an in-training defense method against attribute inference attacks in recommender systems. In addition to the recommendation objective, we define a defensive objective to ensure that the distribution of protected attributes becomes independent of class labels, making users indistinguishable from attribute inference attacks. Specifically, this defensive objective aims to solve a constrained Wasserstein barycenter problem to identify the centroid distribution that makes the attribute indistinguishable while complying with recommendation performance constraints. To optimize our proposed objective, we use optimal transport to align users with the centroid distribution. We conduct extensive experiments on four real-world datasets to evaluate RAID. The experimental results validate the effectiveness of RAID and demonstrate its significant superiority over existing methods in multiple aspects.



## **26. Inferring Communities of Interest in Collaborative Learning-based Recommender Systems**

cs.IR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2306.08929v3) [paper-pdf](http://arxiv.org/pdf/2306.08929v3)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning-based recommender systems, such as those employing Federated Learning (FL) and Gossip Learning (GL), allow users to train models while keeping their history of liked items on their devices. While these methods were seen as promising for enhancing privacy, recent research has shown that collaborative learning can be vulnerable to various privacy attacks. In this paper, we propose a novel attack called Community Inference Attack (CIA), which enables an adversary to identify community members based on a set of target items. What sets CIA apart is its efficiency: it operates at low computational cost by eliminating the need for training surrogate models. Instead, it uses a comparison-based approach, inferring sensitive information by comparing users' models rather than targeting any specific individual model. To evaluate the effectiveness of CIA, we conduct experiments on three real-world recommendation datasets using two recommendation models under both Federated and Gossip-like settings. The results demonstrate that CIA can be up to 10 times more accurate than random guessing. Additionally, we evaluate two mitigation strategies: Differentially Private Stochastic Gradient Descent (DP-SGD) and a Share less policy, which involves sharing fewer, less sensitive model parameters. Our findings suggest that the Share less strategy offers a better privacy-utility trade-off, especially in GL.



## **27. QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models**

cs.CV

Accepted by NAACL 2025 main

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11038v1) [paper-pdf](http://arxiv.org/pdf/2504.11038v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at https://github.com/btzyd/qava.



## **28. Defending Against Frequency-Based Attacks with Diffusion Models**

cs.CV

Conference on Computer Vision and Pattern Recognition Workshops  (CVPRW), 5th Workshop on Adversarial Machine Learning in Computer Vision:  Foundation Models + X

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11034v1) [paper-pdf](http://arxiv.org/pdf/2504.11034v1)

**Authors**: Fatemeh Amerehi, Patrick Healy

**Abstract**: Adversarial training is a common strategy for enhancing model robustness against adversarial attacks. However, it is typically tailored to the specific attack types it is trained on, limiting its ability to generalize to unseen threat models. Adversarial purification offers an alternative by leveraging a generative model to remove perturbations before classification. Since the purifier is trained independently of both the classifier and the threat models, it is better equipped to handle previously unseen attack scenarios. Diffusion models have proven highly effective for noise purification, not only in countering pixel-wise adversarial perturbations but also in addressing non-adversarial data shifts. In this study, we broaden the focus beyond pixel-wise robustness to explore the extent to which purification can mitigate both spectral and spatial adversarial attacks. Our findings highlight its effectiveness in handling diverse distortion patterns across low- to high-frequency regions.



## **29. RF Sensing Security and Malicious Exploitation: A Comprehensive Survey**

cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.10969v1) [paper-pdf](http://arxiv.org/pdf/2504.10969v1)

**Authors**: Mingda Han, Huanqi Yang, Wenhao Li, Weitao Xu, Xiuzhen Cheng, Prasant Mohapatra, Pengfei Hu

**Abstract**: Radio Frequency (RF) sensing technologies have experienced significant growth due to the widespread adoption of RF devices and the Internet of Things (IoT). These technologies enable numerous applications across healthcare, smart homes, industrial automation, and human-computer interaction. However, the non-intrusive and ubiquitous nature of RF sensing - combined with its environmental sensitivity and data dependency - makes these systems inherently vulnerable not only as attack targets, but also as powerful attack vectors. This survey presents a comprehensive analysis of RF sensing security, covering both system-level vulnerabilities - such as signal spoofing, adversarial perturbations, and model poisoning - and the misuse of sensing capabilities for attacks like cross-boundary surveillance, side-channel inference, and semantic privacy breaches. We propose unified threat models to structure these attack vectors and further conduct task-specific vulnerability assessments across key RF sensing applications, identifying their unique attack surfaces and risk profiles. In addition, we systematically review defense strategies across system layers and threat-specific scenarios, incorporating both active and passive paradigms to provide a structured and practical view of protection mechanisms. Compared to prior surveys, our work distinguishes itself by offering a multi-dimensional classification framework based on task type, threat vector, and sensing modality, and by providing fine-grained, scenario-driven analysis that bridges theoretical models and real-world implications. This survey aims to serve as a comprehensive reference for researchers and practitioners seeking to understand, evaluate, and secure the evolving landscape of RF sensing technologies.



## **30. Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters**

cs.LG

Accepted at the 2025 AAAI Conference on Artificial Intelligence  Proceedings

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2501.14122v2) [paper-pdf](http://arxiv.org/pdf/2501.14122v2)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Ricardo Luna Gutierrez, Antonio Guillen

**Abstract**: We present a Reinforcement Learning Platform for Adversarial Black-box untargeted and targeted attacks, RLAB, that allows users to select from various distortion filters to create adversarial examples. The platform uses a Reinforcement Learning agent to add minimum distortion to input images while still causing misclassification by the target model. The agent uses a novel dual-action method to explore the input image at each step to identify sensitive regions for adding distortions while removing noises that have less impact on the target model. This dual action leads to faster and more efficient convergence of the attack. The platform can also be used to measure the robustness of image classification models against specific distortion types. Also, retraining the model with adversarial samples significantly improved robustness when evaluated on benchmark datasets. The proposed platform outperforms state-of-the-art methods in terms of the average number of queries required to cause misclassification. This advances trustworthiness with a positive social impact.



## **31. Verifying Properties of Binary Neural Networks Using Sparse Polynomial Optimization**

cs.LG

25 pages, 2 figures, 8 tables

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2405.17049v2) [paper-pdf](http://arxiv.org/pdf/2405.17049v2)

**Authors**: Jianting Yang, Srećko Ðurašinović, Jean-Bernard Lasserre, Victor Magron, Jun Zhao

**Abstract**: This paper explores methods for verifying the properties of Binary Neural Networks (BNNs), focusing on robustness against adversarial attacks. Despite their lower computational and memory needs, BNNs, like their full-precision counterparts, are also sensitive to input perturbations. Established methods for solving this problem are predominantly based on Satisfiability Modulo Theories and Mixed-Integer Linear Programming techniques, which are characterized by NP complexity and often face scalability issues.   We introduce an alternative approach using Semidefinite Programming relaxations derived from sparse Polynomial Optimization. Our approach, compatible with continuous input space, not only mitigates numerical issues associated with floating-point calculations but also enhances verification scalability through the strategic use of tighter first-order semidefinite relaxations. We demonstrate the effectiveness of our method in verifying robustness against both $\|.\|_\infty$ and $\|.\|_2$-based adversarial attacks.



## **32. CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.10888v1) [paper-pdf](http://arxiv.org/pdf/2504.10888v1)

**Authors**: Jiahuan Long, Wen Yao, Tingsong Jiang, Chao Ma

**Abstract**: Adversarial patches are widely used to evaluate the robustness of object detection systems in real-world scenarios. These patches were initially designed to deceive single-modal detectors (e.g., visible or infrared) and have recently been extended to target visible-infrared dual-modal detectors. However, existing dual-modal adversarial patch attacks have limited attack effectiveness across diverse physical scenarios. To address this, we propose CDUPatch, a universal cross-modal patch attack against visible-infrared object detectors across scales, views, and scenarios. Specifically, we observe that color variations lead to different levels of thermal absorption, resulting in temperature differences in infrared imaging. Leveraging this property, we propose an RGB-to-infrared adapter that maps RGB patches to infrared patches, enabling unified optimization of cross-modal patches. By learning an optimal color distribution on the adversarial patch, we can manipulate its thermal response and generate an adversarial infrared texture. Additionally, we introduce a multi-scale clipping strategy and construct a new visible-infrared dataset, MSDrone, which contains aerial vehicle images in varying scales and perspectives. These data augmentation strategies enhance the robustness of our patch in real-world conditions. Experiments on four benchmark datasets (e.g., DroneVehicle, LLVIP, VisDrone, MSDrone) show that our method outperforms existing patch attacks in the digital domain. Extensive physical tests further confirm strong transferability across scales, views, and scenarios.



## **33. The Sword of Damocles in ViTs: Computational Redundancy Amplifies Adversarial Transferability**

cs.CV

Work in progress. 10 pages. 4 figures

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.10804v1) [paper-pdf](http://arxiv.org/pdf/2504.10804v1)

**Authors**: Jiani Liu, Zhiyuan Wang, Zeliang Zhang, Chao Huang, Susan Liang, Yunlong Tang, Chenliang Xu

**Abstract**: Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.



## **34. Adversarial Prompt Distillation for Vision-Language Models**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.15244v2) [paper-pdf](http://arxiv.org/pdf/2411.15244v2)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.



## **35. Transfer Learning Assisted XgBoost For Adaptable Cyberattack Detection In Battery Packs**

eess.SY

9 pages, 5 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10658v1) [paper-pdf](http://arxiv.org/pdf/2504.10658v1)

**Authors**: Sanchita Ghosh, Tanushree Roy

**Abstract**: Optimal charging of electric vehicle (EVs) depends heavily on reliable sensor measurements from the battery pack to the cloud-controller of the smart charging station. However, an adversary could corrupt the voltage sensor data during transmission, potentially causing local to wide-scale disruptions. Therefore, it is essential to detect sensor cyberattacks in real-time to ensure secure EV charging, and the developed algorithms must be readily adaptable to variations, including pack configurations. To tackle these challenges, we propose adaptable fine-tuning of an XgBoost-based cell-level model using limited pack-level data to use for voltage prediction and residual generation. We used battery cell and pack data from high-fidelity charging experiments in PyBaMM and `liionpack' package to train and test the detection algorithm. The algorithm's performance has been evaluated for two large-format battery packs under sensor swapping and replay attacks. The simulation results also highlight the adaptability and efficacy of our proposed detection algorithm.



## **36. Secure Estimation of Battery Voltage Under Sensor Attacks: A Self-Learning Koopman Approach**

eess.SY

10 pages, 5 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10639v1) [paper-pdf](http://arxiv.org/pdf/2504.10639v1)

**Authors**: Sanchita Ghosh, Tanushree Roy

**Abstract**: Cloud-based battery management system (BMS) requires accurate terminal voltage measurement data to ensure optimal and safe charging of Lithium-ion batteries. Unfortunately, an adversary can corrupt the battery terminal voltage data as it passes from the local-BMS to the cloud-BMS through the communication network, with the objective of under- or over-charging the battery. To ensure accurate terminal voltage data under such malicious sensor attacks, this paper investigates a Koopman-based secure terminal voltage estimation scheme using a two-stage error-compensated self-learning feedback. During the first stage of error correction, the potential Koopman prediction error is estimated to compensate for the error accumulation due to the linear approximation of Koopman operator. The second stage of error compensation aims to recover the error amassing from the higher-order dynamics of the Lithium-ion batteries missed by the self-learning strategy. Specifically, we have proposed two different methods for this second stage error compensation. First, an interpretable empirical correction strategy has been obtained using the open circuit voltage to state-of-charge mapping for the battery. Second, a Gaussian process regression-based data-driven method has been explored. Finally, we demonstrate the efficacy of the proposed secure estimator using both empirical and data-driven corrections.



## **37. Demo: ViolentUTF as An Accessible Platform for Generative AI Red Teaming**

cs.CR

3 pages, 1 figure, 1 table. This is a demo paper for  CyberWarrior2025. Codes and video demo will be shared in later version of  this paper due to embargo requirements

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10603v1) [paper-pdf](http://arxiv.org/pdf/2504.10603v1)

**Authors**: Tam n. Nguyen

**Abstract**: The rapid integration of Generative AI (GenAI) into various applications necessitates robust risk management strategies which includes Red Teaming (RT) - an evaluation method for simulating adversarial attacks. Unfortunately, RT for GenAI is often hindered by technical complexity, lack of user-friendly interfaces, and inadequate reporting features. This paper introduces Violent UTF - an accessible, modular, and scalable platform for GenAI red teaming. Through intuitive interfaces (Web GUI, CLI, API, MCP) powered by LLMs and for LLMs, Violent UTF aims to empower non-technical domain experts and students alongside technical experts, facilitate comprehensive security evaluation by unifying capabilities from RT frameworks like Microsoft PyRIT, Nvidia Garak and its own specialized evaluators. ViolentUTF is being used for evaluating the robustness of a flagship LLM-based product in a large US Government department. It also demonstrates effectiveness in evaluating LLMs' cross-domain reasoning capability between cybersecurity and behavioral psychology.



## **38. Ctrl-Z: Controlling AI Agents via Resampling**

cs.LG

bashcontrol.com

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10374v1) [paper-pdf](http://arxiv.org/pdf/2504.10374v1)

**Authors**: Aryan Bhatt, Cody Rushing, Adam Kaufman, Tyler Tracy, Vasil Georgiev, David Matolcsi, Akbir Khan, Buck Shlegeris

**Abstract**: Control evaluations measure whether monitoring and security protocols for AI systems prevent intentionally subversive AI models from causing harm. Our work presents the first control evaluation performed in an agent environment. We construct BashBench, a dataset of 257 challenging multi-step system administration tasks, and evaluate whether various safety measures can prevent an adversarially constructed AI agent from covertly downloading and executing malicious code in this environment. This multi-step setting introduces new attack and defense dynamics, which we investigate in order to design novel control protocols that prevent safety failures without hindering the ability of non-malicious agents to perform useful work. We introduce a class of control protocols called resample protocols that dynamically take additional samples of certain actions. We find these protocols significantly improve on existing techniques by selectively blocking the AI agent from executing suspicious code and incriminating the agent by generating additional examples of dangerous behavior. We measure the tradeoff between attack prevention and usefulness; our best protocol combines resampling with analysis of previous steps, reducing the success rate of attacks from 58% to 7% at a 5% cost to the performance of a non-malicious agent.



## **39. Universally Composable Commitments with Communicating Malicious Physically Uncloneable Functions**

cs.CR

12 pages + 14 pages Appendix

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10120v1) [paper-pdf](http://arxiv.org/pdf/2504.10120v1)

**Authors**: Lourenço Abecasis, Paulo Mateus, Chrysoula Vlachou

**Abstract**: In this work, we explore the possibility of universally composable (UC)-secure commitments using Physically Uncloneable Functions (PUFs) within a new adversarial model. We introduce the communicating malicious PUFs, i.e. malicious PUFs that can interact with their creator even when not in their possession, obtaining a stronger adversarial model. Prior work [ASIACRYPT 2013, LNCS, vol. 8270, pp. 100-119] proposed a compiler for constructing UC-secure commitments from ideal extractable commitments, and our task would be to adapt the ideal extractable commitment scheme proposed therein to our new model. However, we found an attack and identified a few other issues in that construction, and to address them, we modified the aforementioned ideal extractable commitment scheme and introduced new properties and tools that allow us to rigorously develop and present security proofs in this context. We propose a new UC-secure commitment scheme against adversaries that can only create stateless malicious PUFs which can receive, but not send, information from their creators. Our protocol is more efficient compared to previous proposals, as we have parallelized the ideal extractable commitments within it. The restriction to stateless malicious PUFs is significant, mainly since the protocol from [ASIACRYPT 2013, LNCS, vol. 8270, pp. 100-119] assumes malicious PUFs with unbounded state, thus limiting its applicability. However it is the only way we found to address the issues of the original construction. We hope that in future work this restriction can be lifted, and along the lines of our work, UC-secure commitments with fewer restrictions on both the state and communication can be constructed.



## **40. Undermining Federated Learning Accuracy in EdgeIoT via Variational Graph Auto-Encoders**

cs.LG

7 pages and 6 figures. Accepted in IEEE IWCMC 2025

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10067v1) [paper-pdf](http://arxiv.org/pdf/2504.10067v1)

**Authors**: Kai Li, Shuyan Hu, Bochun Wu, Sai Zou, Wei Ni, Falko Dressler

**Abstract**: EdgeIoT represents an approach that brings together mobile edge computing with Internet of Things (IoT) devices, allowing for data processing close to the data source. Sending source data to a server is bandwidth-intensive and may compromise privacy. Instead, federated learning allows each device to upload a shared machine-learning model update with locally processed data. However, this technique, which depends on aggregating model updates from various IoT devices, is vulnerable to attacks from malicious entities that may inject harmful data into the learning process. This paper introduces a new attack method targeting federated learning in EdgeIoT, known as data-independent model manipulation attack. This attack does not rely on training data from the IoT devices but instead uses an adversarial variational graph auto-encoder (AV-GAE) to create malicious model updates by analyzing benign model updates intercepted during communication. AV-GAE identifies and exploits structural relationships between benign models and their training data features. By manipulating these structural correlations, the attack maximizes the training loss of the federated learning system, compromising its overall effectiveness.



## **41. SnatchML: Hijacking ML models without Training Access**

cs.CR

17 pages, 14 figures, 7 tables

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2406.01708v2) [paper-pdf](http://arxiv.org/pdf/2406.01708v2)

**Authors**: Mahmoud Ghorbel, Halima Bouzidi, Ioan Marius Bilasco, Ihsen Alouani

**Abstract**: Model hijacking can cause significant accountability and security risks since the owner of a hijacked model can be framed for having their model offer illegal or unethical services. Prior works consider model hijacking as a training time attack, whereby an adversary requires full access to the ML model training. In this paper, we consider a stronger threat model for an inference-time hijacking attack, where the adversary has no access to the training phase of the victim model. Our intuition is that ML models, which are typically over-parameterized, might have the capacity to (unintentionally) learn more than the intended task they are trained for. We propose SnatchML, a new training-free model hijacking attack, that leverages the extra capacity learnt by the victim model to infer different tasks that can be semantically related or unrelated to the original one. Our results on models deployed on AWS Sagemaker showed that SnatchML can deliver high accuracy on hijacking tasks. Interestingly, while all previous approaches are limited by the number of classes in the benign task, SnatchML can hijack models for tasks that contain more classes than the original. We explore different methods to mitigate this risk; We propose meta-unlearning, which is designed to help the model unlearn a potentially malicious task while training for the original task. We also provide insights on over-parametrization as a possible inherent factor that facilitates model hijacking, and accordingly, we propose a compression-based countermeasure to counteract this attack. We believe this work offers a previously overlooked perspective on model hijacking attacks, presenting a stronger threat model and higher applicability in real-world contexts.



## **42. Quantifying Privacy Leakage in Split Inference via Fisher-Approximated Shannon Information Analysis**

cs.CR

17pages,14 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10016v1) [paper-pdf](http://arxiv.org/pdf/2504.10016v1)

**Authors**: Ruijun Deng, Zhihui Lu, Qiang Duan

**Abstract**: Split inference (SI) partitions deep neural networks into distributed sub-models, enabling privacy-preserving collaborative learning. Nevertheless, it remains vulnerable to Data Reconstruction Attacks (DRAs), wherein adversaries exploit exposed smashed data to reconstruct raw inputs. Despite extensive research on adversarial attack-defense games, a shortfall remains in the fundamental analysis of privacy risks. This paper establishes a theoretical framework for privacy leakage quantification using information theory, defining it as the adversary's certainty and deriving both average-case and worst-case error bounds. We introduce Fisher-approximated Shannon information (FSInfo), a novel privacy metric utilizing Fisher Information (FI) for operational privacy leakage computation. We empirically show that our privacy metric correlates well with empirical attacks and investigate some of the factors that affect privacy leakage, namely the data distribution, model size, and overfitting.



## **43. Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm**

cs.CV

18 pages, 5 figures. Accepted by Nature Scientific Reports

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2310.13019v5) [paper-pdf](http://arxiv.org/pdf/2310.13019v5)

**Authors**: S. M. Fazle Rabby Labib, Joyanta Jyoti Mondal, Meem Arafat Manab, Xi Xiao, Sarfaraz Newaz

**Abstract**: The susceptibility of deep neural networks (DNNs) to adversarial attacks undermines their reliability across numerous applications, underscoring the necessity for an in-depth exploration of these vulnerabilities and the formulation of robust defense strategies. The DeepFool algorithm by Moosavi-Dezfooli et al. (2016) represents a pivotal step in identifying minimal perturbations required to induce misclassification of input images. Nonetheless, its generic methodology falls short in scenarios necessitating targeted interventions. Additionally, previous research studies have predominantly concentrated on the success rate of attacks without adequately addressing the consequential distortion of images, the maintenance of image quality, or the confidence threshold required for misclassification. To bridge these gaps, we introduce the Enhanced Targeted DeepFool (ET DeepFool) algorithm, an evolution of DeepFool that not only facilitates the specification of desired misclassification targets but also incorporates a configurable minimum confidence score. Our empirical investigations demonstrate the superiority of this refined approach in maintaining the integrity of images and minimizing perturbations across a variety of DNN architectures. Unlike previous iterations, such as the Targeted DeepFool by Gajjar et al. (2022), our method grants unparalleled control over the perturbation process, enabling precise manipulation of model responses. Preliminary outcomes reveal that certain models, including AlexNet and the advanced Vision Transformer, display commendable robustness to such manipulations. This discovery of varying levels of model robustness, as unveiled through our confidence level adjustments, could have far-reaching implications for the field of image recognition. Our code is available at https://github.com/FazleLabib/et_deepfool.



## **44. Let the Noise Speak: Harnessing Noise for a Unified Defense Against Adversarial and Backdoor Attacks**

cs.LG

20 pages, 9 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2406.13073v2) [paper-pdf](http://arxiv.org/pdf/2406.13073v2)

**Authors**: Md Hasan Shahriar, Ning Wang, Naren Ramakrishnan, Y. Thomas Hou, Wenjing Lou

**Abstract**: The exponential adoption of machine learning (ML) is propelling the world into a future of distributed and intelligent automation and data-driven solutions. However, the proliferation of malicious data manipulation attacks against ML, namely adversarial and backdoor attacks, jeopardizes its reliability in safety-critical applications. The existing detection methods are attack-specific and built upon some strong assumptions, limiting them in diverse practical scenarios. Thus, motivated by the need for a more robust, unified, and attack-agnostic defense mechanism, we first investigate the shared traits of adversarial and backdoor attacks. Based on our observation, we propose NoiSec, a reconstruction-based intrusion detection system that brings a novel perspective by shifting focus from the reconstructed input to the reconstruction noise itself, which is the foundational root cause of such malicious data alterations. NoiSec disentangles the noise from the test input, extracts the underlying features from the noise, and leverages them to recognize systematic malicious manipulation. Our comprehensive evaluation of NoiSec demonstrates its high effectiveness across various datasets, including basic objects, natural scenes, traffic signs, medical images, spectrogram-based audio data, and wireless sensing against five state-of-the-art adversarial attacks and three backdoor attacks under challenging evaluation conditions. NoiSec demonstrates strong detection performance in both white-box and black-box adversarial attack scenarios, significantly outperforming the closest baseline models, particularly in an adaptive attack setting. We will provide the code for future baseline comparison. Our code and artifacts are publicly available at https://github.com/shahriar0651/NoiSec.



## **45. An Investigation of Large Language Models and Their Vulnerabilities in Spam Detection**

cs.CR

10 pages; presented at HotSoS'2025 as a work in progress paper

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.09776v1) [paper-pdf](http://arxiv.org/pdf/2504.09776v1)

**Authors**: Qiyao Tang, Xiangyang Li

**Abstract**: Spam messages continue to present significant challenges to digital users, cluttering inboxes and posing security risks. Traditional spam detection methods, including rules-based, collaborative, and machine learning approaches, struggle to keep up with the rapidly evolving tactics employed by spammers. This project studies new spam detection systems that leverage Large Language Models (LLMs) fine-tuned with spam datasets. More importantly, we want to understand how LLM-based spam detection systems perform under adversarial attacks that purposefully modify spam emails and data poisoning attacks that exploit the differences between the training data and the massages in detection, to which traditional machine learning models are shown to be vulnerable. This experimentation employs two LLM models of GPT2 and BERT and three spam datasets of Enron, LingSpam, and SMSspamCollection for extensive training and testing tasks. The results show that, while they can function as effective spam filters, the LLM models are susceptible to the adversarial and data poisoning attacks. This research provides very useful insights for future applications of LLM models for information security.



## **46. NODE-AdvGAN: Improving the transferability and perceptual similarity of adversarial examples by dynamic-system-driven adversarial generative model**

cs.LG

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2412.03539v2) [paper-pdf](http://arxiv.org/pdf/2412.03539v2)

**Authors**: Xinheng Xie, Yue Wu, Cuiyu He

**Abstract**: Understanding adversarial examples is crucial for improving model robustness, as they introduce imperceptible perturbations to deceive models. Effective adversarial examples, therefore, offer the potential to train more robust models by eliminating model singularities. We propose NODE-AdvGAN, a novel approach that treats adversarial generation as a continuous process and employs a Neural Ordinary Differential Equation (NODE) to simulate generator dynamics. By mimicking the iterative nature of traditional gradient-based methods, NODE-AdvGAN generates smoother and more precise perturbations that preserve high perceptual similarity when added to benign images. We also propose a new training strategy, NODE-AdvGAN-T, which enhances transferability in black-box attacks by tuning the noise parameters during training. Experiments demonstrate that NODE-AdvGAN and NODE-AdvGAN-T generate more effective adversarial examples that achieve higher attack success rates while preserving better perceptual quality than baseline models.



## **47. Mitigating Many-Shot Jailbreaking**

cs.LG

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09604v1) [paper-pdf](http://arxiv.org/pdf/2504.09604v1)

**Authors**: Christopher M. Ackerman, Nina Panickssery

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a ``fake'' assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the ``fake'' assistant. In this work, we probe the effectiveness of different fine tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and we show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training.



## **48. ControlNET: A Firewall for RAG-based LLM System**

cs.CR

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09593v1) [paper-pdf](http://arxiv.org/pdf/2504.09593v1)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.



## **49. Secure Filtering against Spatio-Temporal False Data Attacks under Asynchronous Sampling**

eess.SY

9 pages and 6 figures. arXiv admin note: text overlap with  arXiv:2303.17514

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2411.19765v2) [paper-pdf](http://arxiv.org/pdf/2411.19765v2)

**Authors**: Zishuo Li, Anh Tung Nguyen, André M. H. Teixeira, Yilin Mo, Karl H. Johansson

**Abstract**: This paper addresses the secure state estimation problem for continuous linear time-invariant systems with non-periodic and asynchronous sampled measurements, where the sensors need to transmit not only measurements but also sampling time-stamps to the fusion center. This measurement and communication setup is well-suited for operating large-scale control systems and, at the same time, introduces new vulnerabilities that can be exploited by adversaries through (i) manipulation of measurements, (ii) manipulation of time-stamps, (iii) elimination of measurements, (iv) generation of completely new false measurements, or a combination of these attacks. To mitigate these attacks, we propose a decentralized estimation algorithm in which each sensor maintains its local state estimate asynchronously based on its measurements. The local states are synchronized through time prediction and fused after time-stamp alignment. In the absence of attacks, state estimates are proven to recover the optimal Kalman estimates by solving a weighted least square problem. In the presence of attacks, solving this weighted least square problem with the aid of $\ell_1$ regularization provides secure state estimates with uniformly bounded error under an observability redundancy assumption. The effectiveness of the proposed algorithm is demonstrated using a benchmark example of the IEEE 14-bus system.



## **50. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

cs.CR

17 pages, 6 figures, 9 tables

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09466v1) [paper-pdf](http://arxiv.org/pdf/2504.09466v1)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.



