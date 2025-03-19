# Latest Adversarial Attack Papers
**update at 2025-03-19 10:38:18**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. VGFL-SA: Vertical Graph Federated Learning Structure Attack Based on Contrastive Learning**

cs.LG

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2502.16793v2) [paper-pdf](http://arxiv.org/pdf/2502.16793v2)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks (GNNs) have gained attention for their ability to learn representations from graph data. Due to privacy concerns and conflicts of interest that prevent clients from directly sharing graph data with one another, Vertical Graph Federated Learning (VGFL) frameworks have been developed. Recent studies have shown that VGFL is vulnerable to adversarial attacks that degrade performance. However, it is a common problem that client nodes are often unlabeled in the realm of VGFL. Consequently, the existing attacks, which rely on the availability of labeling information to obtain gradients, are inherently constrained in their applicability. This limitation precludes their deployment in practical, real-world environments. To address the above problems, we propose a novel graph adversarial attack against VGFL, referred to as VGFL-SA, to degrade the performance of VGFL by modifying the local clients structure without using labels. Specifically, VGFL-SA uses a contrastive learning method to complete the attack before the local clients are trained. VGFL-SA first accesses the graph structure and node feature information of the poisoned clients, and generates the contrastive views by node-degree-based edge augmentation and feature shuffling augmentation. Then, VGFL-SA uses the shared graph encoder to get the embedding of each view, and the gradients of the adjacency matrices are obtained by the contrastive function. Finally, perturbed edges are generated using gradient modification rules. We validated the performance of VGFL-SA by performing a node classification task on real-world datasets, and the results show that VGFL-SA achieves good attack effectiveness and transferability.



## **2. Unveiling the Role of Randomization in Multiclass Adversarial Classification: Insights from Graph Theory**

cs.LG

9 pages (main), 30 in total. Camera-ready version, accepted at  AISTATS 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14299v1) [paper-pdf](http://arxiv.org/pdf/2503.14299v1)

**Authors**: Lucas Gnecco-Heredia, Matteo Sammut, Muni Sreenivas Pydi, Rafael Pinot, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Randomization as a mean to improve the adversarial robustness of machine learning models has recently attracted significant attention. Unfortunately, much of the theoretical analysis so far has focused on binary classification, providing only limited insights into the more complex multiclass setting. In this paper, we take a step toward closing this gap by drawing inspiration from the field of graph theory. Our analysis focuses on discrete data distributions, allowing us to cast the adversarial risk minimization problems within the well-established framework of set packing problems. By doing so, we are able to identify three structural conditions on the support of the data distribution that are necessary for randomization to improve robustness. Furthermore, we are able to construct several data distributions where (contrarily to binary classification) switching from a deterministic to a randomized solution significantly reduces the optimal adversarial risk. These findings highlight the crucial role randomization can play in enhancing robustness to adversarial attacks in multiclass classification.



## **3. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

cs.CV

Under review

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2405.18770v2) [paper-pdf](http://arxiv.org/pdf/2405.18770v2)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. Our experiments show that MAT can effectively be applied to different VL models and tasks to improve adversarial robustness, outperforming previous efforts. Our code will be made public upon acceptance.



## **4. XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants**

cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14281v1) [paper-pdf](http://arxiv.org/pdf/2503.14281v1)

**Authors**: Adam Štorek, Mukur Gupta, Noopur Bhatt, Aditya Gupta, Janie Kim, Prashast Srivastava, Suman Jana

**Abstract**: AI coding assistants are widely used for tasks like code generation, bug detection, and comprehension. These tools now require large and complex contexts, automatically sourced from various origins$\unicode{x2014}$across files, projects, and contributors$\unicode{x2014}$forming part of the prompt fed to underlying LLMs. This automatic context-gathering introduces new vulnerabilities, allowing attackers to subtly poison input to compromise the assistant's outputs, potentially generating vulnerable code, overlooking flaws, or introducing critical errors. We propose a novel attack, Cross-Origin Context Poisoning (XOXO), that is particularly challenging to detect as it relies on adversarial code modifications that are semantically equivalent. Traditional program analysis techniques struggle to identify these correlations since the semantics of the code remain correct, making it appear legitimate. This allows attackers to manipulate code assistants into producing incorrect outputs, including vulnerabilities or backdoors, while shifting the blame to the victim developer or tester. We introduce a novel, task-agnostic black-box attack algorithm GCGS that systematically searches the transformation space using a Cayley Graph, achieving an 83.09% attack success rate on average across five tasks and eleven models, including GPT-4o and Claude 3.5 Sonnet v2 used by many popular AI coding assistants. Furthermore, existing defenses, including adversarial fine-tuning, are ineffective against our attack, underscoring the need for new security measures in LLM-powered coding tools.



## **5. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

cs.CL

Accepted to the NAACL PrivateNLP 2025 Workshop

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2407.21630v2) [paper-pdf](http://arxiv.org/pdf/2407.21630v2)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduces the accuracy of attackers while preserving utility. We make our code and models publicly available.



## **6. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.04833v2) [paper-pdf](http://arxiv.org/pdf/2503.04833v2)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.



## **7. Survey of Adversarial Robustness in Multimodal Large Language Models**

cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](http://arxiv.org/pdf/2503.13962v1)

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.



## **8. Make the Most of Everything: Further Considerations on Disrupting Diffusion-based Customization**

cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13945v1) [paper-pdf](http://arxiv.org/pdf/2503.13945v1)

**Authors**: Long Tang, Dengpan Ye, Sirun Chen, Xiuwen Shi, Yunna Lv, Ziyi Liu

**Abstract**: The fine-tuning technique for text-to-image diffusion models facilitates image customization but risks privacy breaches and opinion manipulation. Current research focuses on prompt- or image-level adversarial attacks for anti-customization, yet it overlooks the correlation between these two levels and the relationship between internal modules and inputs. This hinders anti-customization performance in practical threat scenarios. We propose Dual Anti-Diffusion (DADiff), a two-stage adversarial attack targeting diffusion customization, which, for the first time, integrates the adversarial prompt-level attack into the generation process of image-level adversarial examples. In stage 1, we generate prompt-level adversarial vectors to guide the subsequent image-level attack. In stage 2, besides conducting the end-to-end attack on the UNet model, we disrupt its self- and cross-attention modules, aiming to break the correlations between image pixels and align the cross-attention results computed using instance prompts and adversarial prompt vectors within the images. Furthermore, we introduce a local random timestep gradient ensemble strategy, which updates adversarial perturbations by integrating random gradients from multiple segmented timesets. Experimental results on various mainstream facial datasets demonstrate 10%-30% improvements in cross-prompt, keyword mismatch, cross-model, and cross-mechanism anti-customization with DADiff compared to existing methods.



## **9. GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack**

cs.CV

This article has been accepted for publication at ICLR 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.12827v2) [paper-pdf](http://arxiv.org/pdf/2503.12827v2)

**Authors**: Md Farhamdur Reza, Richeng Jin, Tianfu Wu, Huaiyu Dai

**Abstract**: Existing score-based adversarial attacks mainly focus on crafting $top$-1 adversarial examples against classifiers with single-label classification. Their attack success rate and query efficiency are often less than satisfactory, particularly under small perturbation requirements; moreover, the vulnerability of classifiers with multi-label learning is yet to be studied. In this paper, we propose a comprehensive surrogate free score-based attack, named \b geometric \b score-based \b black-box \b attack (GSBA$^K$), to craft adversarial examples in an aggressive $top$-$K$ setting for both untargeted and targeted attacks, where the goal is to change the $top$-$K$ predictions of the target classifier. We introduce novel gradient-based methods to find a good initial boundary point to attack. Our iterative method employs novel gradient estimation techniques, particularly effective in $top$-$K$ setting, on the decision boundary to effectively exploit the geometry of the decision boundary. Additionally, GSBA$^K$ can be used to attack against classifiers with $top$-$K$ multi-label learning. Extensive experimental results on ImageNet and PASCAL VOC datasets validate the effectiveness of GSBA$^K$ in crafting $top$-$K$ adversarial examples.



## **10. Securing Virtual Reality Experiences: Unveiling and Tackling Cybersickness Attacks with Explainable AI**

cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13419v1) [paper-pdf](http://arxiv.org/pdf/2503.13419v1)

**Authors**: Ripan Kumar Kundu, Matthew Denton, Genova Mongalo, Prasad Calyam, Khaza Anuarul Hoque

**Abstract**: The synergy between virtual reality (VR) and artificial intelligence (AI), specifically deep learning (DL)-based cybersickness detection models, has ushered in unprecedented advancements in immersive experiences by automatically detecting cybersickness severity and adaptively various mitigation techniques, offering a smooth and comfortable VR experience. While this DL-enabled cybersickness detection method provides promising solutions for enhancing user experiences, it also introduces new risks since these models are vulnerable to adversarial attacks; a small perturbation of the input data that is visually undetectable to human observers can fool the cybersickness detection model and trigger unexpected mitigation, thus disrupting user immersive experiences (UIX) and even posing safety risks. In this paper, we present a new type of VR attack, i.e., a cybersickness attack, which successfully stops the triggering of cybersickness mitigation by fooling DL-based cybersickness detection models and dramatically hinders the UIX. Next, we propose a novel explainable artificial intelligence (XAI)-guided cybersickness attack detection framework to detect such attacks in VR to ensure UIX and a comfortable VR experience. We evaluate the proposed attack and the detection framework using two state-of-the-art open-source VR cybersickness datasets: Simulation 2021 and Gameplay dataset. Finally, to verify the effectiveness of our proposed method, we implement the attack and the XAI-based detection using a testbed with a custom-built VR roller coaster simulation with an HTC Vive Pro Eye headset and perform a user study. Our study shows that such an attack can dramatically hinder the UIX. However, our proposed XAI-guided cybersickness attack detection can successfully detect cybersickness attacks and trigger the proper mitigation, effectively reducing VR cybersickness.



## **11. On the Byzantine-Resilience of Distillation-Based Federated Learning**

cs.LG

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2402.12265v3) [paper-pdf](http://arxiv.org/pdf/2402.12265v3)

**Authors**: Christophe Roux, Max Zimmer, Sebastian Pokutta

**Abstract**: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and instead communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process. Based on these insights, we introduce two new byzantine attacks and demonstrate their ability to break existing byzantine-resilient methods. Additionally, we propose a novel defence method which enhances the byzantine resilience of KD-based FL algorithms. Finally, we provide a general framework to obfuscate attacks, making them significantly harder to detect, thereby improving their effectiveness. Our findings serve as an important building block in the analysis of byzantine FL, contributing through the development of new attacks and new defence mechanisms, further advancing the robustness of KD-based FL algorithms.



## **12. How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark**

eess.IV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12990v1) [paper-pdf](http://arxiv.org/pdf/2503.12990v1)

**Authors**: Roba Al Majzoub, Hashmat Malik, Muzammal Naseer, Zaigham Zaheer, Tariq Mahmood, Salman Khan, Fahad Khan

**Abstract**: Recently, histopathology vision-language foundation models (VLMs) have gained popularity due to their enhanced performance and generalizability across different downstream tasks. However, most existing histopathology benchmarks are either unimodal or limited in terms of diversity of clinical tasks, organs, and acquisition instruments, as well as their partial availability to the public due to patient data privacy. As a consequence, there is a lack of comprehensive evaluation of existing histopathology VLMs on a unified benchmark setting that better reflects a wide range of clinical scenarios. To address this gap, we introduce HistoVL, a fully open-source comprehensive benchmark comprising images acquired using up to 11 various acquisition tools that are paired with specifically crafted captions by incorporating class names and diverse pathology descriptions. Our Histo-VL includes 26 organs, 31 cancer types, and a wide variety of tissue obtained from 14 heterogeneous patient cohorts, totaling more than 5 million patches obtained from over 41K WSIs viewed under various magnification levels. We systematically evaluate existing histopathology VLMs on Histo-VL to simulate diverse tasks performed by experts in real-world clinical scenarios. Our analysis reveals interesting findings, including large sensitivity of most existing histopathology VLMs to textual changes with a drop in balanced accuracy of up to 25% in tasks such as Metastasis detection, low robustness to adversarial attacks, as well as improper calibration of models evident through high ECE values and low model prediction confidence, all of which can affect their clinical implementation.



## **13. Distributed Black-box Attack: Do Not Overestimate Black-box Attacks**

cs.LG

Accepted by ICLR Workshop, 2025

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2210.16371v5) [paper-pdf](http://arxiv.org/pdf/2210.16371v5)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: As cloud computing becomes pervasive, deep learning models are deployed on cloud servers and then provided as APIs to end users. However, black-box adversarial attacks can fool image classification models without access to model structure and weights. Recent studies have reported attack success rates of over 95% with fewer than 1,000 queries. Then the question arises: whether black-box attacks have become a real threat against cloud APIs? To shed some light on this, our research indicates that black-box attacks are not as effective against cloud APIs as proposed in research papers due to several common mistakes that overestimate the efficiency of black-box attacks. To avoid similar mistakes, we conduct black-box attacks directly on cloud APIs rather than local models.



## **14. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12793v1) [paper-pdf](http://arxiv.org/pdf/2503.12793v1)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.



## **15. Algebraic Adversarial Attacks on Explainability Models**

cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12683v1) [paper-pdf](http://arxiv.org/pdf/2503.12683v1)

**Authors**: Lachlan Simpson, Federico Costanza, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

**Abstract**: Classical adversarial attacks are phrased as a constrained optimisation problem. Despite the efficacy of a constrained optimisation approach to adversarial attacks, one cannot trace how an adversarial point was generated. In this work, we propose an algebraic approach to adversarial attacks and study the conditions under which one can generate adversarial examples for post-hoc explainability models. Phrasing neural networks in the framework of geometric deep learning, algebraic adversarial attacks are constructed through analysis of the symmetry groups of neural networks. Algebraic adversarial examples provide a mathematically tractable approach to adversarial examples. We validate our approach of algebraic adversarial examples on two well-known and one real-world dataset.



## **16. Provably Reliable Conformal Prediction Sets in the Presence of Data Poisoning**

cs.LG

Accepted at ICLR 2025 (Spotlight)

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2410.09878v4) [paper-pdf](http://arxiv.org/pdf/2410.09878v4)

**Authors**: Yan Scholten, Stephan Günnemann

**Abstract**: Conformal prediction provides model-agnostic and distribution-free uncertainty quantification through prediction sets that are guaranteed to include the ground truth with any user-specified probability. Yet, conformal prediction is not reliable under poisoning attacks where adversaries manipulate both training and calibration data, which can significantly alter prediction sets in practice. As a solution, we propose reliable prediction sets (RPS): the first efficient method for constructing conformal prediction sets with provable reliability guarantees under poisoning. To ensure reliability under training poisoning, we introduce smoothed score functions that reliably aggregate predictions of classifiers trained on distinct partitions of the training data. To ensure reliability under calibration poisoning, we construct multiple prediction sets, each calibrated on distinct subsets of the calibration data. We then aggregate them into a majority prediction set, which includes a class only if it appears in a majority of the individual sets. Both proposed aggregations mitigate the influence of datapoints in the training and calibration data on the final prediction set. We experimentally validate our approach on image classification tasks, achieving strong reliability while maintaining utility and preserving coverage on clean data. Overall, our approach represents an important step towards more trustworthy uncertainty quantification in the presence of data poisoning.



## **17. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

cs.CR

14 pages

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.02986v3) [paper-pdf](http://arxiv.org/pdf/2503.02986v3)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.



## **18. GAN-Based Single-Stage Defense for Traffic Sign Classification Under Adversarial Patch Attack**

cs.CV

This work has been submitted to the IEEE Transactions on Intelligent  Transportation Systems (T-ITS) for possible publication

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12567v1) [paper-pdf](http://arxiv.org/pdf/2503.12567v1)

**Authors**: Abyad Enan, Mashrur Chowdhury

**Abstract**: Computer Vision plays a critical role in ensuring the safe navigation of autonomous vehicles (AVs). An AV perception module is responsible for capturing and interpreting the surrounding environment to facilitate safe navigation. This module enables AVs to recognize traffic signs, traffic lights, and various road users. However, the perception module is vulnerable to adversarial attacks, which can compromise their accuracy and reliability. One such attack is the adversarial patch attack (APA), a physical attack in which an adversary strategically places a specially crafted sticker on an object to deceive object classifiers. In APA, an adversarial patch is positioned on a target object, leading the classifier to misidentify it. Such an APA can cause AVs to misclassify traffic signs, leading to catastrophic incidents. To enhance the security of an AV perception system against APAs, this study develops a Generative Adversarial Network (GAN)-based single-stage defense strategy for traffic sign classification. This approach is tailored to defend against APAs on different classes of traffic signs without prior knowledge of a patch's design. This study found this approach to be effective against patches of varying sizes. Our experimental analysis demonstrates that the defense strategy presented in this paper improves the classifier's accuracy under APA conditions by up to 80.8% and enhances overall classification accuracy for all the traffic signs considered in this study by 58%, compared to a classifier without any defense mechanism. Our defense strategy is model-agnostic, making it applicable to any traffic sign classifier, regardless of the underlying classification model.



## **19. On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis**

cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2502.13191v2) [paper-pdf](http://arxiv.org/pdf/2502.13191v2)

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at https://anonymous.4open.science/r/MIA_SNN-3610.



## **20. Towards Privacy-Preserving Data-Driven Education: The Potential of Federated Learning**

cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.13550v1) [paper-pdf](http://arxiv.org/pdf/2503.13550v1)

**Authors**: Mohammad Khalil, Ronas Shakya, Qinyi Liu

**Abstract**: The increasing adoption of data-driven applications in education such as in learning analytics and AI in education has raised significant privacy and data protection concerns. While these challenges have been widely discussed in previous works, there are still limited practical solutions. Federated learning has recently been discoursed as a promising privacy-preserving technique, yet its application in education remains scarce. This paper presents an experimental evaluation of federated learning for educational data prediction, comparing its performance to traditional non-federated approaches. Our findings indicate that federated learning achieves comparable predictive accuracy. Furthermore, under adversarial attacks, federated learning demonstrates greater resilience compared to non-federated settings. We summarise that our results reinforce the value of federated learning as a potential approach for balancing predictive performance and privacy in educational contexts.



## **21. CARNet: Collaborative Adversarial Resilience for Robust Underwater Image Enhancement and Perception**

cs.CV

13 pages, 13 figures

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2309.01102v2) [paper-pdf](http://arxiv.org/pdf/2309.01102v2)

**Authors**: Zengxi Zhang, Zeru Shi, Zhiying Jiang, Jinyuan Liu

**Abstract**: Due to the uneven absorption of different light wavelengths in aquatic environments, underwater images suffer from low visibility and clear color deviations. With the advancement of autonomous underwater vehicles, extensive research has been conducted on learning-based underwater enhancement algorithms. These works can generate visually pleasing enhanced images and mitigate the adverse effects of degraded images on subsequent perception tasks. However, learning-based methods are susceptible to the inherent fragility of adversarial attacks, causing significant disruption in enhanced results. In this work, we introduce a collaborative adversarial resilience network, dubbed CARNet, for underwater image enhancement and subsequent detection tasks. Concretely, we first introduce an invertible network with strong perturbation-perceptual abilities to isolate attacks from underwater images, preventing interference with visual quality enhancement and perceptual tasks. Furthermore, an attack pattern discriminator is introduced to adaptively identify and eliminate various types of attacks. Additionally, we propose a bilevel attack optimization strategy to heighten the robustness of the network against different types of attacks under the collaborative adversarial training of vision-driven and perception-driven attacks. Extensive experiments demonstrate that the proposed method outputs visually appealing enhancement images and performs an average 6.71% higher detection mAP than state-of-the-art methods.



## **22. Augmented Adversarial Trigger Learning**

cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12339v1) [paper-pdf](http://arxiv.org/pdf/2503.12339v1)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs.



## **23. Training-Free Mitigation of Adversarial Attacks on Deep Learning-Based MRI Reconstruction**

cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2501.01908v2) [paper-pdf](http://arxiv.org/pdf/2501.01908v2)

**Authors**: Mahdi Saberi, Chi Zhang, Mehmet Akcakaya

**Abstract**: Deep learning (DL) methods, especially those based on physics-driven DL, have become the state-of-the-art for reconstructing sub-sampled magnetic resonance imaging (MRI) data. However, studies have shown that these methods are susceptible to small adversarial input perturbations, or attacks, resulting in major distortions in the output images. Various strategies have been proposed to reduce the effects of these attacks, but they require retraining and may lower reconstruction quality for non-perturbed/clean inputs. In this work, we propose a novel approach for mitigating adversarial attacks on MRI reconstruction models without any retraining. Our framework is based on the idea of cyclic measurement consistency. The output of the model is mapped to another set of MRI measurements for a different sub-sampling pattern, and this synthesized data is reconstructed with the same model. Intuitively, without an attack, the second reconstruction is expected to be consistent with the first, while with an attack, disruptions are present. A novel objective function is devised based on this idea, which is minimized within a small ball around the attack input for mitigation. Experimental results show that our method substantially reduces the impact of adversarial perturbations across different datasets, attack types/strengths and PD-DL networks, and qualitatively and quantitatively outperforms conventional mitigation methods that involve retraining. Finally, we extend our mitigation method to two important practical scenarios: a blind setup, where the attack strength or algorithm is not known to the end user; and an adaptive attack setup, where the attacker has full knowledge of the defense strategy. Our approach remains effective in both cases.



## **24. Multi-Agent Systems Execute Arbitrary Malicious Code**

cs.CR

30 pages, 5 figures, 8 tables

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2503.12188v1) [paper-pdf](http://arxiv.org/pdf/2503.12188v1)

**Authors**: Harold Triedman, Rishi Jha, Vitaly Shmatikov

**Abstract**: Multi-agent systems coordinate LLM-based agents to perform tasks on users' behalf. In real-world applications, multi-agent systems will inevitably interact with untrusted inputs, such as malicious Web content, files, email attachments, etc.   Using several recently proposed multi-agent frameworks as concrete examples, we demonstrate that adversarial content can hijack control and communication within the system to invoke unsafe agents and functionalities. This results in a complete security breach, up to execution of arbitrary malicious code on the user's device and/or exfiltration of sensitive data from the user's containerized environment. We show that control-flow hijacking attacks succeed even if the individual agents are not susceptible to direct or indirect prompt injection, and even if they refuse to perform harmful actions.



## **25. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

cs.CR

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2407.20361v3) [paper-pdf](http://arxiv.org/pdf/2407.20361v3)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models -- Stack model, VisualPhishNet, and Phishpedia -- against PhishOracle-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a multimodal large language model (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether PhishOracle-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources will be made publicly available on GitHub.



## **26. Robust Dataset Distillation by Matching Adversarial Trajectories**

cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2503.12069v1) [paper-pdf](http://arxiv.org/pdf/2503.12069v1)

**Authors**: Wei Lai, Tianyu Ding, ren dongdong, Lei Wang, Jing Huo, Yang Gao, Wenbin Li

**Abstract**: Dataset distillation synthesizes compact datasets that enable models to achieve performance comparable to training on the original large-scale datasets. However, existing distillation methods overlook the robustness of the model, resulting in models that are vulnerable to adversarial attacks when trained on distilled data. To address this limitation, we introduce the task of ``robust dataset distillation", a novel paradigm that embeds adversarial robustness into the synthetic datasets during the distillation process. We propose Matching Adversarial Trajectories (MAT), a method that integrates adversarial training into trajectory-based dataset distillation. MAT incorporates adversarial samples during trajectory generation to obtain robust training trajectories, which are then used to guide the distillation process. As experimentally demonstrated, even through natural training on our distilled dataset, models can achieve enhanced adversarial robustness while maintaining competitive accuracy compared to existing distillation methods. Our work highlights robust dataset distillation as a new and important research direction and provides a strong baseline for future research to bridge the gap between efficient training and adversarial robustness.



## **27. On Minimizing Adversarial Counterfactual Error in Adversarial RL**

cs.LG

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2406.04724v3) [paper-pdf](http://arxiv.org/pdf/2406.04724v3)

**Authors**: Roman Belaire, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Deep Reinforcement Learning (DRL) policies are highly susceptible to adversarial noise in observations, which poses significant risks in safety-critical scenarios. The challenge inherent to adversarial perturbations is that by altering the information observed by the agent, the state becomes only partially observable. Existing approaches address this by either enforcing consistent actions across nearby states or maximizing the worst-case value within adversarially perturbed observations. However, the former suffers from performance degradation when attacks succeed, while the latter tends to be overly conservative, leading to suboptimal performance in benign settings. We hypothesize that these limitations stem from their failing to account for partial observability directly. To this end, we introduce a novel objective called Adversarial Counterfactual Error (ACoE), defined on the beliefs about the true state and balancing value optimization with robustness. To make ACoE scalable in model-free settings, we propose the theoretically-grounded surrogate objective Cumulative-ACoE (C-ACoE). Our empirical evaluations on standard benchmarks (MuJoCo, Atari, and Highway) demonstrate that our method significantly outperforms current state-of-the-art approaches for addressing adversarial RL challenges, offering a promising direction for improving robustness in DRL under adversarial conditions. Our code is available at https://github.com/romanbelaire/acoe-robust-rl.



## **28. Robust and Efficient Adversarial Defense in SNNs via Image Purification and Joint Detection**

cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2404.17092v2) [paper-pdf](http://arxiv.org/pdf/2404.17092v2)

**Authors**: Weiran Chen, Qi Xu

**Abstract**: Spiking Neural Networks (SNNs) aim to bridge the gap between neuroscience and machine learning by emulating the structure of the human nervous system. However, like convolutional neural networks, SNNs are vulnerable to adversarial attacks. To tackle the challenge, we propose a biologically inspired methodology to enhance the robustness of SNNs, drawing insights from the visual masking effect and filtering theory. First, an end-to-end SNN-based image purification model is proposed to defend against adversarial attacks, including a noise extraction network and a non-blind denoising network. The former network extracts noise features from noisy images, while the latter component employs a residual U-Net structure to reconstruct high-quality noisy images and generate clean images. Simultaneously, a multi-level firing SNN based on Squeeze-and-Excitation Network is introduced to improve the robustness of the classifier. Crucially, the proposed image purification network serves as a pre-processing module, avoiding modifications to classifiers. Unlike adversarial training, our method is highly flexible and can be seamlessly integrated with other defense strategies. Experimental results on various datasets demonstrate that the proposed methodology outperforms state-of-the-art baselines in terms of defense effectiveness, training time, and resource consumption.



## **29. A Framework for Evaluating Emerging Cyberattack Capabilities of AI**

cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11917v1) [paper-pdf](http://arxiv.org/pdf/2503.11917v1)

**Authors**: Mikel Rodriguez, Raluca Ada Popa, Four Flynn, Lihao Liang, Allan Dafoe, Anna Wang

**Abstract**: As frontier models become more capable, the community has attempted to evaluate their ability to enable cyberattacks. Performing a comprehensive evaluation and prioritizing defenses are crucial tasks in preparing for AGI safely. However, current cyber evaluation efforts are ad-hoc, with no systematic reasoning about the various phases of attacks, and do not provide a steer on how to use targeted defenses. In this work, we propose a novel approach to AI cyber capability evaluation that (1) examines the end-to-end attack chain, (2) helps to identify gaps in the evaluation of AI threats, and (3) helps defenders prioritize targeted mitigations and conduct AI-enabled adversary emulation to support red teaming. To achieve these goals, we propose adapting existing cyberattack chain frameworks to AI systems. We analyze over 12,000 instances of real-world attempts to use AI in cyberattacks catalogued by Google's Threat Intelligence Group. Using this analysis, we curate a representative collection of seven cyberattack chain archetypes and conduct a bottleneck analysis to identify areas of potential AI-driven cost disruption. Our evaluation benchmark consists of 50 new challenges spanning different phases of cyberattacks. Based on this, we devise targeted cybersecurity model evaluations, report on the potential for AI to amplify offensive cyber capabilities across specific attack phases, and conclude with recommendations on prioritizing defenses. In all, we consider this to be the most comprehensive AI cyber risk evaluation framework published so far.



## **30. Order Fairness Evaluation of DAG-based ledgers**

cs.CR

17 double-column pages with 9 pages dedicated to references and  appendices, 22 figures, 13 of which are in the appendices

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2502.17270v2) [paper-pdf](http://arxiv.org/pdf/2502.17270v2)

**Authors**: Erwan Mahe, Sara Tucci-Piergiovanni

**Abstract**: Order fairness in distributed ledgers refers to properties that relate the order in which transactions are sent or received to the order in which they are eventually finalized, i.e., totally ordered. The study of such properties is relatively new and has been especially stimulated by the rise of Maximal Extractable Value (MEV) attacks in blockchain environments. Indeed, in many classical blockchain protocols, leaders are responsible for selecting the transactions to be included in blocks, which creates a clear vulnerability and opportunity for transaction order manipulation.   Unlike blockchains, DAG-based ledgers allow participants in the network to independently propose blocks, which are then arranged as vertices of a directed acyclic graph. Interestingly, leaders in DAG-based ledgers are elected only after the fact, once transactions are already part of the graph, to determine their total order. In other words, transactions are not chosen by single leaders; instead, they are collectively validated by the nodes, and leaders are only elected to establish an ordering. This approach intuitively reduces the risk of transaction manipulation and enhances fairness.   In this paper, we aim to quantify the capability of DAG-based ledgers to achieve order fairness. To this end, we define new variants of order fairness adapted to DAG-based ledgers and evaluate the impact of an adversary capable of compromising a limited number of nodes (below the one-third threshold) to reorder transactions. We analyze how often our order fairness properties are violated under different network conditions and parameterizations of the DAG algorithm, depending on the adversary's power.   Our study shows that DAG-based ledgers are still vulnerable to reordering attacks, as an adversary can coordinate a minority of Byzantine nodes to manipulate the DAG's structure.



## **31. Enhancing Resiliency of Sketch-based Security via LSB Sharing-based Dynamic Late Merging**

cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11777v1) [paper-pdf](http://arxiv.org/pdf/2503.11777v1)

**Authors**: Seungsam Yang, Seyed Mohammad Mehdi Mirnajafizadeh, Sian Kim, Rhongho Jang, DaeHun Nyang

**Abstract**: With the exponentially growing Internet traffic, sketch data structure with a probabilistic algorithm has been expected to be an alternative solution for non-compromised (non-selective) security monitoring. While facilitating counting within a confined memory space, the sketch's memory efficiency and accuracy were further pushed to their limit through finer-grained and dynamic control of constrained memory space to adapt to the data stream's inherent skewness (i.e., Zipf distribution), namely small counters with extensions. In this paper, we unveil a vulnerable factor of the small counter design by introducing a new sketch-oriented attack, which threatens a stream of state-of-the-art sketches and their security applications. With the root cause analyses, we propose Siamese Counter with enhanced adversarial resiliency and verified feasibility with extensive experimental and theoretical analyses. Under a sketch pollution attack, Siamese Counter delivers 47% accurate results than a state-of-the-art scheme, and demonstrates up to 82% more accurate estimation under normal measurement scenarios.



## **32. Making Every Step Effective: Jailbreaking Large Vision-Language Models Through Hierarchical KV Equalization**

cs.CV

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11750v1) [paper-pdf](http://arxiv.org/pdf/2503.11750v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Jun Liu, Muhao Chen, Zi Huang, Yujun Cai

**Abstract**: In the realm of large vision-language models (LVLMs), adversarial jailbreak attacks serve as a red-teaming approach to identify safety vulnerabilities of these models and their associated defense mechanisms. However, we identify a critical limitation: not every adversarial optimization step leads to a positive outcome, and indiscriminately accepting optimization results at each step may reduce the overall attack success rate. To address this challenge, we introduce HKVE (Hierarchical Key-Value Equalization), an innovative jailbreaking framework that selectively accepts gradient optimization results based on the distribution of attention scores across different layers, ensuring that every optimization step positively contributes to the attack. Extensive experiments demonstrate HKVE's significant effectiveness, achieving attack success rates of 75.08% on MiniGPT4, 85.84% on LLaVA and 81.00% on Qwen-VL, substantially outperforming existing methods by margins of 20.43\%, 21.01\% and 26.43\% respectively. Furthermore, making every step effective not only leads to an increase in attack success rate but also allows for a reduction in the number of iterations, thereby lowering computational costs. Warning: This paper contains potentially harmful example data.



## **33. Are Deep Speech Denoising Models Robust to Adversarial Noise?**

cs.SD

13 pages, 5 figures

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11627v1) [paper-pdf](http://arxiv.org/pdf/2503.11627v1)

**Authors**: Will Schwarzer, Philip S. Thomas, Andrea Fanelli, Xiaoyu Liu

**Abstract**: Deep noise suppression (DNS) models enjoy widespread use throughout a variety of high-stakes speech applications. However, in this paper, we show that four recent DNS models can each be reduced to outputting unintelligible gibberish through the addition of imperceptible adversarial noise. Furthermore, our results show the near-term plausibility of targeted attacks, which could induce models to output arbitrary utterances, and over-the-air attacks. While the success of these attacks varies by model and setting, and attacks appear to be strongest when model-specific (i.e., white-box and non-transferable), our results highlight a pressing need for practical countermeasures in DNS systems.



## **34. Tit-for-Tat: Safeguarding Large Vision-Language Models Against Jailbreak Attacks via Adversarial Defense**

cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11619v1) [paper-pdf](http://arxiv.org/pdf/2503.11619v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Ming-Hsuan Yang, Jun Liu, Chengcheng Tang, Zi Huang, Yujun Cai

**Abstract**: Deploying large vision-language models (LVLMs) introduces a unique vulnerability: susceptibility to malicious attacks via visual inputs. However, existing defense methods suffer from two key limitations: (1) They solely focus on textual defenses, fail to directly address threats in the visual domain where attacks originate, and (2) the additional processing steps often incur significant computational overhead or compromise model performance on benign tasks. Building on these insights, we propose ESIII (Embedding Security Instructions Into Images), a novel methodology for transforming the visual space from a source of vulnerability into an active defense mechanism. Initially, we embed security instructions into defensive images through gradient-based optimization, obtaining security instructions in the visual dimension. Subsequently, we integrate security instructions from visual and textual dimensions with the input query. The collaboration between security instructions from different dimensions ensures comprehensive security protection. Extensive experiments demonstrate that our approach effectively fortifies the robustness of LVLMs against such attacks while preserving their performance on standard benign tasks and incurring an imperceptible increase in time costs.



## **35. Feasibility of Randomized Detector Tuning for Attack Impact Mitigation**

eess.SY

8 pages, 4 figures, submitted to CDC 2025

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11417v1) [paper-pdf](http://arxiv.org/pdf/2503.11417v1)

**Authors**: Sribalaji C. Anand, Kamil Hassan, Henrik Sandberg

**Abstract**: This paper considers the problem of detector tuning against false data injection attacks. In particular, we consider an adversary injecting false sensor data to maximize the state deviation of the plant, referred to as impact, whilst being stealthy. To minimize the impact of stealthy attacks, inspired by moving target defense, the operator randomly switches the detector thresholds. In this paper, we theoretically derive the sufficient (and in some cases necessary) conditions under which the impact of stealthy attacks can be made smaller with randomized switching of detector thresholds compared to static thresholds. We establish the conditions for the stateless ($\chi^2$) and the stateful (CUSUM) detectors. The results are illustrated through numerical examples.



## **36. Hiding Local Manipulations on SAR Images: a Counter-Forensic Attack**

cs.CV

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2407.07041v2) [paper-pdf](http://arxiv.org/pdf/2407.07041v2)

**Authors**: Sara Mandelli, Edoardo Daniele Cannas, Paolo Bestagini, Stefano Tebaldini, Stefano Tubaro

**Abstract**: The vast accessibility of Synthetic Aperture Radar (SAR) images through online portals has propelled the research across various fields. This widespread use and easy availability have unfortunately made SAR data susceptible to malicious alterations, such as local editing applied to the images for inserting or covering the presence of sensitive targets. Vulnerability is further emphasized by the fact that most SAR products, despite their original complex nature, are often released as amplitude-only information, allowing even inexperienced attackers to edit and easily alter the pixel content. To contrast malicious manipulations, in the last years the forensic community has begun to dig into the SAR manipulation issue, proposing detectors that effectively localize the tampering traces in amplitude images. Nonetheless, in this paper we demonstrate that an expert practitioner can exploit the complex nature of SAR data to obscure any signs of manipulation within a locally altered amplitude image. We refer to this approach as a counter-forensic attack. To achieve the concealment of manipulation traces, the attacker can simulate a re-acquisition of the manipulated scene by the SAR system that initially generated the pristine image. In doing so, the attacker can obscure any evidence of manipulation, making it appear as if the image was legitimately produced by the system. This attack has unique features that make it both highly generalizable and relatively easy to apply. First, it is a black-box attack, meaning it is not designed to deceive a specific forensic detector. Furthermore, it does not require a training phase and is not based on adversarial operations. We assess the effectiveness of the proposed counter-forensic approach across diverse scenarios, examining various manipulation operations.



## **37. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

cs.IT

16 pages, 23 figures

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2402.10686v3) [paper-pdf](http://arxiv.org/pdf/2402.10686v3)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.



## **38. SmartShards: Churn-Tolerant Continuously Available Distributed Ledger**

cs.DC

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11077v1) [paper-pdf](http://arxiv.org/pdf/2503.11077v1)

**Authors**: Joseph Oglio, Mikhail Nesterenko, Gokarna Sharma

**Abstract**: We present SmartShards: a new sharding algorithm for improving Byzantine tolerance and churn resistance in blockchains. Our algorithm places a peer in multiple shards to create an overlap. This simplifies cross-shard communication and shard membership management. We describe SmartShards, prove it correct and evaluate its performance.   We propose several SmartShards extensions: defense against a slowly adaptive adversary, combining transactions into blocks, fortification against the join/leave attack.



## **39. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

cs.CR

15 pages; This update was mistakenly uploaded as a new manuscript on  arXiv (2503.06529). The wrong submission has now been withdrawn, and we  replace the old one here

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2411.14243v2) [paper-pdf](http://arxiv.org/pdf/2411.14243v2)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control.



## **40. Efficient Reachability Analysis for Convolutional Neural Networks Using Hybrid Zonotopes**

math.OC

Accepted by 2025 American Control Conference (ACC). 8 pages, 1 figure

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10840v1) [paper-pdf](http://arxiv.org/pdf/2503.10840v1)

**Authors**: Yuhao Zhang, Xiangru Xu

**Abstract**: Feedforward neural networks are widely used in autonomous systems, particularly for control and perception tasks within the system loop. However, their vulnerability to adversarial attacks necessitates formal verification before deployment in safety-critical applications. Existing set propagation-based reachability analysis methods for feedforward neural networks often struggle to achieve both scalability and accuracy. This work presents a novel set-based approach for computing the reachable sets of convolutional neural networks. The proposed method leverages a hybrid zonotope representation and an efficient neural network reduction technique, providing a flexible trade-off between computational complexity and approximation accuracy. Numerical examples are presented to demonstrate the effectiveness of the proposed approach.



## **41. Attacking Multimodal OS Agents with Malicious Image Patches**

cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10809v1) [paper-pdf](http://arxiv.org/pdf/2503.10809v1)

**Authors**: Lukas Aichberger, Alasdair Paren, Yarin Gal, Philip Torr, Adel Bibi

**Abstract**: Recent advances in operating system (OS) agents enable vision-language models to interact directly with the graphical user interface of an OS. These multimodal OS agents autonomously perform computer-based tasks in response to a single prompt via application programming interfaces (APIs). Such APIs typically support low-level operations, including mouse clicks, keyboard inputs, and screenshot captures. We introduce a novel attack vector: malicious image patches (MIPs) that have been adversarially perturbed so that, when captured in a screenshot, they cause an OS agent to perform harmful actions by exploiting specific APIs. For instance, MIPs embedded in desktop backgrounds or shared on social media can redirect an agent to a malicious website, enabling further exploitation. These MIPs generalise across different user requests and screen layouts, and remain effective for multiple OS agents. The existence of such attacks highlights critical security vulnerabilities in OS agents, which should be carefully addressed before their widespread adoption.



## **42. Byzantine-Resilient Federated Learning via Distributed Optimization**

cs.LG

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10792v1) [paper-pdf](http://arxiv.org/pdf/2503.10792v1)

**Authors**: Yufei Xia, Wenrui Yu, Qiongxiu Li

**Abstract**: Byzantine attacks present a critical challenge to Federated Learning (FL), where malicious participants can disrupt the training process, degrade model accuracy, and compromise system reliability. Traditional FL frameworks typically rely on aggregation-based protocols for model updates, leaving them vulnerable to sophisticated adversarial strategies. In this paper, we demonstrate that distributed optimization offers a principled and robust alternative to aggregation-centric methods. Specifically, we show that the Primal-Dual Method of Multipliers (PDMM) inherently mitigates Byzantine impacts by leveraging its fault-tolerant consensus mechanism. Through extensive experiments on three datasets (MNIST, FashionMNIST, and Olivetti), under various attack scenarios including bit-flipping and Gaussian noise injection, we validate the superior resilience of distributed optimization protocols. Compared to traditional aggregation-centric approaches, PDMM achieves higher model utility, faster convergence, and improved stability. Our results highlight the effectiveness of distributed optimization in defending against Byzantine threats, paving the way for more secure and resilient federated learning systems.



## **43. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.



## **44. Hierarchical Self-Supervised Adversarial Training for Robust Vision Models in Histopathology**

cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10629v1) [paper-pdf](http://arxiv.org/pdf/2503.10629v1)

**Authors**: Hashmat Shadab Malik, Shahina Kunhimon, Muzammal Naseer, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Adversarial attacks pose significant challenges for vision models in critical fields like healthcare, where reliability is essential. Although adversarial training has been well studied in natural images, its application to biomedical and microscopy data remains limited. Existing self-supervised adversarial training methods overlook the hierarchical structure of histopathology images, where patient-slide-patch relationships provide valuable discriminative signals. To address this, we propose Hierarchical Self-Supervised Adversarial Training (HSAT), which exploits these properties to craft adversarial examples using multi-level contrastive learning and integrate it into adversarial training for enhanced robustness. We evaluate HSAT on multiclass histopathology dataset OpenSRH and the results show that HSAT outperforms existing methods from both biomedical and natural image domains. HSAT enhances robustness, achieving an average gain of 54.31% in the white-box setting and reducing performance drops to 3-4% in the black-box setting, compared to 25-30% for the baseline. These results set a new benchmark for adversarial training in this domain, paving the way for more robust models. Our Code for training and evaluation is available at https://github.com/HashmatShadab/HSAT.



## **45. Towards Class-wise Robustness Analysis**

cs.LG

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2411.19853v2) [paper-pdf](http://arxiv.org/pdf/2411.19853v2)

**Authors**: Tejaswini Medi, Julia Grabinski, Margret Keuper

**Abstract**: While being very successful in solving many downstream tasks, the application of deep neural networks is limited in real-life scenarios because of their susceptibility to domain shifts such as common corruptions, and adversarial attacks. The existence of adversarial examples and data corruption significantly reduces the performance of deep classification models. Researchers have made strides in developing robust neural architectures to bolster decisions of deep classifiers. However, most of these works rely on effective adversarial training methods, and predominantly focus on overall model robustness, disregarding class-wise differences in robustness, which are critical. Exploiting weakly robust classes is a potential avenue for attackers to fool the image recognition models. Therefore, this study investigates class-to-class biases across adversarially trained robust classification models to understand their latent space structures and analyze their strong and weak class-wise properties. We further assess the robustness of classes against common corruptions and adversarial attacks, recognizing that class vulnerability extends beyond the number of correct classifications for a specific class. We find that the number of false positives of classes as specific target classes significantly impacts their vulnerability to attacks. Through our analysis on the Class False Positive Score, we assess a fair evaluation of how susceptible each class is to misclassification.



## **46. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.



## **47. I Can Tell Your Secrets: Inferring Privacy Attributes from Mini-app Interaction History in Super-apps**

cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10239v1) [paper-pdf](http://arxiv.org/pdf/2503.10239v1)

**Authors**: Yifeng Cai, Ziqi Zhang, Mengyu Yao, Junlin Liu, Xiaoke Zhao, Xinyi Fu, Ruoyu Li, Zhe Li, Xiangqun Chen, Yao Guo, Ding Li

**Abstract**: Super-apps have emerged as comprehensive platforms integrating various mini-apps to provide diverse services. While super-apps offer convenience and enriched functionality, they can introduce new privacy risks. This paper reveals a new privacy leakage source in super-apps: mini-app interaction history, including mini-app usage history (Mini-H) and operation history (Op-H). Mini-H refers to the history of mini-apps accessed by users, such as their frequency and categories. Op-H captures user interactions within mini-apps, including button clicks, bar drags, and image views. Super-apps can naturally collect these data without instrumentation due to the web-based feature of mini-apps. We identify these data types as novel and unexplored privacy risks through a literature review of 30 papers and an empirical analysis of 31 super-apps. We design a mini-app interaction history-oriented inference attack (THEFT), to exploit this new vulnerability. Using THEFT, the insider threats within the low-privilege business department of the super-app vendor acting as the adversary can achieve more than 95.5% accuracy in inferring privacy attributes of over 16.1% of users. THEFT only requires a small training dataset of 200 users from public breached databases on the Internet. We also engage with super-app vendors and a standards association to increase industry awareness and commitment to protect this data. Our contributions are significant in identifying overlooked privacy risks, demonstrating the effectiveness of a new attack, and influencing industry practices toward better privacy protection in the super-app ecosystem.



## **48. Robustness Tokens: Towards Adversarial Robustness of Transformers**

cs.LG

This paper has been accepted for publication at the European  Conference on Computer Vision (ECCV), 2024

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10191v1) [paper-pdf](http://arxiv.org/pdf/2503.10191v1)

**Authors**: Brian Pulfer, Yury Belousov, Slava Voloshynovskiy

**Abstract**: Recently, large pre-trained foundation models have become widely adopted by machine learning practitioners for a multitude of tasks. Given that such models are publicly available, relying on their use as backbone models for downstream tasks might result in high vulnerability to adversarial attacks crafted with the same public model. In this work, we propose Robustness Tokens, a novel approach specific to the transformer architecture that fine-tunes a few additional private tokens with low computational requirements instead of tuning model parameters as done in traditional adversarial training. We show that Robustness Tokens make Vision Transformer models significantly more robust to white-box adversarial attacks while also retaining the original downstream performances.



## **49. Can't Slow me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices**

cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.02171v2) [paper-pdf](http://arxiv.org/pdf/2412.02171v2)

**Authors**: Tianyi Wang, Zichen Wang, Cong Wang, Yuanchao Shu, Ruilong Deng, Peng Cheng, Jiming Chen

**Abstract**: Object detection is a fundamental enabler for many real-time downstream applications such as autonomous driving, augmented reality and supply chain management. However, the algorithmic backbone of neural networks is brittle to imperceptible perturbations in the system inputs, which were generally known as misclassifying attacks. By targeting the real-time processing capability, a new class of latency attacks are reported recently. They exploit new attack surfaces in object detectors by creating a computational bottleneck in the post-processing module, that leads to cascading failure and puts the real-time downstream tasks at risks. In this work, we take an initial attempt to defend against this attack via background-attentive adversarial training that is also cognizant of the underlying hardware capabilities. We first draw system-level connections between latency attack and hardware capacity across heterogeneous GPU devices. Based on the particular adversarial behaviors, we utilize objectness loss as a proxy and build background attention into the adversarial training pipeline, and achieve a reasonable balance between clean and robust accuracy. The extensive experiments demonstrate the defense effectiveness of restoring real-time processing capability from $13$ FPS to $43$ FPS on Jetson Orin NX, with a better trade-off between the clean and robust accuracy.



## **50. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

cs.CV

Accepted to ECCV 2024 (Oral), Project Page:  https://PDCL-Attack.github.io

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2407.20657v2) [paper-pdf](http://arxiv.org/pdf/2407.20657v2)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.



