# Latest Adversarial Attack Papers
**update at 2025-05-30 15:22:54**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents**

cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23559v1) [paper-pdf](http://arxiv.org/pdf/2505.23559v1)

**Authors**: Kunlun Zhu, Jiaxun Zhang, Ziheng Qi, Nuoxing Shang, Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You

**Abstract**: Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}



## **2. SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.14667v2) [paper-pdf](http://arxiv.org/pdf/2410.14667v2)

**Authors**: Peimeng Guan, Mark A. Davenport

**Abstract**: Inverse problems aim to reconstruct unseen data from corrupted or perturbed measurements. While most work focuses on improving reconstruction quality, generalization accuracy and robustness are equally important, especially for safety-critical applications. Model-based architectures (MBAs), such as loop unrolling methods, are considered more interpretable and achieve better reconstructions. Empirical evidence suggests that MBAs are more robust to perturbations than black-box solvers, but the accuracy-robustness tradeoff in MBAs remains underexplored. In this work, we propose a simple yet effective training scheme for MBAs, called SGD jittering, which injects noise iteration-wise during reconstruction. We theoretically demonstrate that SGD jittering not only generalizes better than the standard mean squared error training but is also more robust to average-case attacks. We validate SGD jittering using denoising toy examples, seismic deconvolution, and single-coil MRI reconstruction. Both SGD jittering and its SPGD extension yield cleaner reconstructions for out-of-distribution data and demonstrates enhanced robustness against adversarial attacks.



## **3. TRAP: Targeted Redirecting of Agentic Preferences**

cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23518v1) [paper-pdf](http://arxiv.org/pdf/2505.23518v1)

**Authors**: Hangoo Kang, Jehyeok Yeon, Gagandeep Singh

**Abstract**: Autonomous agentic AI systems powered by vision-language models (VLMs) are rapidly advancing toward real-world deployment, yet their cross-modal reasoning capabilities introduce new attack surfaces for adversarial manipulation that exploit semantic reasoning across modalities. Existing adversarial attacks typically rely on visible pixel perturbations or require privileged model or environment access, making them impractical for stealthy, real-world exploitation. We introduce TRAP, a generative adversarial framework that manipulates the agent's decision-making using diffusion-based semantic injections. Our method combines negative prompt-based degradation with positive semantic optimization, guided by a Siamese semantic network and layout-aware spatial masking. Without requiring access to model internals, TRAP produces visually natural images yet induces consistent selection biases in agentic AI systems. We evaluate TRAP on the Microsoft Common Objects in Context (COCO) dataset, building multi-candidate decision scenarios. Across these scenarios, TRAP achieves a 100% attack success rate on leading models, including LLaVA-34B, Gemma3, and Mistral-3.1, significantly outperforming baselines such as SPSA, Bandit, and standard diffusion approaches. These results expose a critical vulnerability: Autonomous agents can be consistently misled through human-imperceptible cross-modal manipulations. These findings highlight the need for defense strategies beyond pixel-level robustness to address semantic vulnerabilities in cross-modal decision-making.



## **4. Hijacking Large Language Models via Adversarial In-Context Learning**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2311.09948v3) [paper-pdf](http://arxiv.org/pdf/2311.09948v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Prashant Khanduri, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the preconditioned prompts. Despite its promising performance, crafted adversarial attacks pose a notable threat to the robustness of LLMs. Existing attacks are either easy to detect, require a trigger in user input, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable prompt injection attack against ICL, aiming to hijack LLMs to generate the target output or elicit harmful responses. In our threat model, the hacker acts as a model publisher who leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos via prompt injection. We also propose effective defense strategies using a few shots of clean demos, enhancing the robustness of LLMs during ICL. Extensive experimental results across various classification and jailbreak tasks demonstrate the effectiveness of the proposed attack and defense strategies. This work highlights the significant security vulnerabilities of LLMs during ICL and underscores the need for further in-depth studies.



## **5. Learning to Poison Large Language Models for Downstream Manipulation**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2402.13459v3) [paper-pdf](http://arxiv.org/pdf/2402.13459v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where the adversary inserts backdoor triggers into training data to manipulate outputs. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the supervised fine-tuning (SFT) process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various language model tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during SFT of LLMs and the necessity of safeguarding LLMs against data poisoning attacks.



## **6. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.11647v2) [paper-pdf](http://arxiv.org/pdf/2502.11647v2)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.



## **7. Robustness-Congruent Adversarial Training for Secure Machine Learning Model Updates**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2402.17390v2) [paper-pdf](http://arxiv.org/pdf/2402.17390v2)

**Authors**: Daniele Angioni, Luca Demetrio, Maura Pintor, Luca Oneto, Davide Anguita, Battista Biggio, Fabio Roli

**Abstract**: Machine-learning models demand periodic updates to improve their average accuracy, exploiting novel architectures and additional data. However, a newly updated model may commit mistakes the previous model did not make. Such misclassifications are referred to as negative flips, experienced by users as a regression of performance. In this work, we show that this problem also affects robustness to adversarial examples, hindering the development of secure model update practices. In particular, when updating a model to improve its adversarial robustness, previously ineffective adversarial attacks on some inputs may become successful, causing a regression in the perceived security of the system. We propose a novel technique, named robustness-congruent adversarial training, to address this issue. It amounts to fine-tuning a model with adversarial training, while constraining it to retain higher robustness on the samples for which no adversarial example was found before the update. We show that our algorithm and, more generally, learning with non-regression constraints, provides a theoretically-grounded framework to train consistent estimators. Our experiments on robust models for computer vision confirm that both accuracy and robustness, even if improved after model update, can be affected by negative flips, and our robustness-congruent adversarial training can mitigate the problem, outperforming competing baseline methods.



## **8. Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models**

cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23404v1) [paper-pdf](http://arxiv.org/pdf/2505.23404v1)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin

**Abstract**: Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)



## **9. Adversarial Semantic and Label Perturbation Attack for Pedestrian Attribute Recognition**

cs.CV

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23313v1) [paper-pdf](http://arxiv.org/pdf/2505.23313v1)

**Authors**: Weizhe Kong, Xiao Wang, Ruichong Gao, Chenglong Li, Yu Zhang, Xing Yang, Yaowei Wang, Jin Tang

**Abstract**: Pedestrian Attribute Recognition (PAR) is an indispensable task in human-centered research and has made great progress in recent years with the development of deep neural networks. However, the potential vulnerability and anti-interference ability have still not been fully explored. To bridge this gap, this paper proposes the first adversarial attack and defense framework for pedestrian attribute recognition. Specifically, we exploit both global- and patch-level attacks on the pedestrian images, based on the pre-trained CLIP-based PAR framework. It first divides the input pedestrian image into non-overlapping patches and embeds them into feature embeddings using a projection layer. Meanwhile, the attribute set is expanded into sentences using prompts and embedded into attribute features using a pre-trained CLIP text encoder. A multi-modal Transformer is adopted to fuse the obtained vision and text tokens, and a feed-forward network is utilized for attribute recognition. Based on the aforementioned PAR framework, we adopt the adversarial semantic and label-perturbation to generate the adversarial noise, termed ASL-PAR. We also design a semantic offset defense strategy to suppress the influence of adversarial attacks. Extensive experiments conducted on both digital domains (i.e., PETA, PA100K, MSP60K, RAPv2) and physical domains fully validated the effectiveness of our proposed adversarial attack and defense strategies for the pedestrian attribute recognition. The source code of this paper will be released on https://github.com/Event-AHU/OpenPAR.



## **10. Disrupting Vision-Language Model-Driven Navigation Services via Adversarial Object Fusion**

cs.CR

Under review

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23266v1) [paper-pdf](http://arxiv.org/pdf/2505.23266v1)

**Authors**: Chunlong Xie, Jialing He, Shangwei Guo, Jiacheng Wang, Shudong Zhang, Tianwei Zhang, Tao Xiang

**Abstract**: We present Adversarial Object Fusion (AdvOF), a novel attack framework targeting vision-and-language navigation (VLN) agents in service-oriented environments by generating adversarial 3D objects. While foundational models like Large Language Models (LLMs) and Vision Language Models (VLMs) have enhanced service-oriented navigation systems through improved perception and decision-making, their integration introduces vulnerabilities in mission-critical service workflows. Existing adversarial attacks fail to address service computing contexts, where reliability and quality-of-service (QoS) are paramount. We utilize AdvOF to investigate and explore the impact of adversarial environments on the VLM-based perception module of VLN agents. In particular, AdvOF first precisely aggregates and aligns the victim object positions in both 2D and 3D space, defining and rendering adversarial objects. Then, we collaboratively optimize the adversarial object with regularization between the adversarial and victim object across physical properties and VLM perceptions. Through assigning importance weights to varying views, the optimization is processed stably and multi-viewedly by iterative fusions from local updates and justifications. Our extensive evaluations demonstrate AdvOF can effectively degrade agent performance under adversarial conditions while maintaining minimal interference with normal navigation tasks. This work advances the understanding of service security in VLM-powered navigation systems, providing computational foundations for robust service composition in physical-world deployments.



## **11. Are You Using Reliable Graph Prompts? Trojan Prompt Attacks on Graph Neural Networks**

cs.LG

To be appeared in KDD 2025

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.13974v2) [paper-pdf](http://arxiv.org/pdf/2410.13974v2)

**Authors**: Minhua Lin, Zhiwei Zhang, Enyan Dai, Zongyu Wu, Yilong Wang, Xiang Zhang, Suhang Wang

**Abstract**: Graph Prompt Learning (GPL) has been introduced as a promising approach that uses prompts to adapt pre-trained GNN models to specific downstream tasks without requiring fine-tuning of the entire model. Despite the advantages of GPL, little attention has been given to its vulnerability to backdoor attacks, where an adversary can manipulate the model's behavior by embedding hidden triggers. Existing graph backdoor attacks rely on modifying model parameters during training, but this approach is impractical in GPL as GNN encoder parameters are frozen after pre-training. Moreover, downstream users may fine-tune their own task models on clean datasets, further complicating the attack. In this paper, we propose TGPA, a backdoor attack framework designed specifically for GPL. TGPA injects backdoors into graph prompts without modifying pre-trained GNN encoders and ensures high attack success rates and clean accuracy. To address the challenge of model fine-tuning by users, we introduce a finetuning-resistant poisoning approach that maintains the effectiveness of the backdoor even after downstream model adjustments. Extensive experiments on multiple datasets under various settings demonstrate the effectiveness of TGPA in compromising GPL models with fixed GNN encoders.



## **12. The Meeseeks Mesh: Spatially Consistent 3D Adversarial Objects for BEV Detector**

cs.CV

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.22499v2) [paper-pdf](http://arxiv.org/pdf/2505.22499v2)

**Authors**: Aixuan Li, Mochu Xiang, Jing Zhang, Yuchao Dai

**Abstract**: 3D object detection is a critical component in autonomous driving systems. It allows real-time recognition and detection of vehicles, pedestrians and obstacles under varying environmental conditions. Among existing methods, 3D object detection in the Bird's Eye View (BEV) has emerged as the mainstream framework. To guarantee a safe, robust and trustworthy 3D object detection, 3D adversarial attacks are investigated, where attacks are placed in 3D environments to evaluate the model performance, e.g. putting a film on a car, clothing a pedestrian. The vulnerability of 3D object detection models to 3D adversarial attacks serves as an important indicator to evaluate the robustness of the model against perturbations. To investigate this vulnerability, we generate non-invasive 3D adversarial objects tailored for real-world attack scenarios. Our method verifies the existence of universal adversarial objects that are spatially consistent across time and camera views. Specifically, we employ differentiable rendering techniques to accurately model the spatial relationship between adversarial objects and the target vehicle. Furthermore, we introduce an occlusion-aware module to enhance visual consistency and realism under different viewpoints. To maintain attack effectiveness across multiple frames, we design a BEV spatial feature-guided optimization strategy. Experimental results demonstrate that our approach can reliably suppress vehicle predictions from state-of-the-art 3D object detectors, serving as an important tool to test robustness of 3D object detection models before deployment. Moreover, the generated adversarial objects exhibit strong generalization capabilities, retaining its effectiveness at various positions and distances in the scene.



## **13. Fooling the Watchers: Breaking AIGC Detectors via Semantic Prompt Attacks**

cs.CV

9 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23192v1) [paper-pdf](http://arxiv.org/pdf/2505.23192v1)

**Authors**: Run Hao, Peng Ying

**Abstract**: The rise of text-to-image (T2I) models has enabled the synthesis of photorealistic human portraits, raising serious concerns about identity misuse and the robustness of AIGC detectors. In this work, we propose an automated adversarial prompt generation framework that leverages a grammar tree structure and a variant of the Monte Carlo tree search algorithm to systematically explore the semantic prompt space. Our method generates diverse, controllable prompts that consistently evade both open-source and commercial AIGC detectors. Extensive experiments across multiple T2I models validate its effectiveness, and the approach ranked first in a real-world adversarial AIGC detection competition. Beyond attack scenarios, our method can also be used to construct high-quality adversarial datasets, providing valuable resources for training and evaluating more robust AIGC detection and defense systems.



## **14. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

cs.CL

arXiv admin note: text overlap with arXiv:2407.14644

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2412.16359v3) [paper-pdf](http://arxiv.org/pdf/2412.16359v3)

**Authors**: Nilanjana Das, Edward Raff, Aman Chadha, Manas Gaur

**Abstract**: As the AI systems become deeply embedded in social media platforms, we've uncovered a concerning security vulnerability that goes beyond traditional adversarial attacks. It becomes important to assess the risks of LLMs before the general public use them on social media platforms to avoid any adverse impacts. Unlike obvious nonsensical text strings that safety systems can easily catch, our work reveals that human-readable situation-driven adversarial full-prompts that leverage situational context are effective but much harder to detect. We found that skilled attackers can exploit the vulnerabilities in open-source and proprietary LLMs to make a malicious user query safe for LLMs, resulting in generating a harmful response. This raises an important question about the vulnerabilities of LLMs. To measure the robustness against human-readable attacks, which now present a potent threat, our research makes three major contributions. First, we developed attacks that use movie scripts as situational contextual frameworks, creating natural-looking full-prompts that trick LLMs into generating harmful content. Second, we developed a method to transform gibberish adversarial text into readable, innocuous content that still exploits vulnerabilities when used within the full-prompts. Finally, we enhanced the AdvPrompter framework with p-nucleus sampling to generate diverse human-readable adversarial texts that significantly improve attack effectiveness against models like GPT-3.5-Turbo-0125 and Gemma-7b. Our findings show that these systems can be manipulated to operate beyond their intended ethical boundaries when presented with seemingly normal prompts that contain hidden adversarial elements. By identifying these vulnerabilities, we aim to drive the development of more robust safety mechanisms that can withstand sophisticated attacks in real-world applications.



## **15. One Prompt to Verify Your Models: Black-Box Text-to-Image Models Verification via Non-Transferable Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.22725v4) [paper-pdf](http://arxiv.org/pdf/2410.22725v4)

**Authors**: Ji Guo, Wenbo Jiang, Rui Zhang, Guoming Lu, Hongwei Li

**Abstract**: Recently, various types of Text-to-Image (T2I) models have emerged (such as DALL-E and Stable Diffusion), and showing their advantages in different aspects. Therefore, some third-party service platforms collect different model interfaces and provide cheaper API services and more flexibility in T2I model selections. However, this also raises a new security concern: Are these third-party services truly offering the models they claim?   To answer this question, we first define the concept of T2I model verification, which aims to determine whether a black-box target model is identical to a given white-box reference T2I model. After that, we propose VerifyPrompt, which performs T2I model verification through a special designed verify prompt. Intuitionally, the verify prompt is an adversarial prompt for the target model without transferability for other models. It makes the target model generate a specific image while making other models produce entirely different images. Specifically, VerifyPrompt utilizes the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize the cosine similarity of a prompt's text encoding, generating verify prompts. Finally, by computing the CLIP-text similarity scores between the prompts the generated images, VerifyPrompt can determine whether the target model aligns with the reference model. Experimental results demonstrate that VerifyPrompt consistently achieves over 90\% accuracy across various T2I models, confirming its effectiveness in practical model platforms (such as Hugging Face).



## **16. Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates**

cs.CL

ACL 2025 Main. Code is released at  https://vision.snu.ac.kr/projects/mac

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22943v1) [paper-pdf](http://arxiv.org/pdf/2505.22943v1)

**Authors**: Jaewoo Ahn, Heeseung Yun, Dayoon Ko, Gunhee Kim

**Abstract**: While pre-trained multimodal representations (e.g., CLIP) have shown impressive capabilities, they exhibit significant compositional vulnerabilities leading to counterintuitive judgments. We introduce Multimodal Adversarial Compositionality (MAC), a benchmark that leverages large language models (LLMs) to generate deceptive text samples to exploit these vulnerabilities across different modalities and evaluates them through both sample-wise attack success rate and group-wise entropy-based diversity. To improve zero-shot methods, we propose a self-training approach that leverages rejection-sampling fine-tuning with diversity-promoting filtering, which enhances both attack success rate and sample diversity. Using smaller language models like Llama-3.1-8B, our approach demonstrates superior performance in revealing compositional vulnerabilities across various multimodal representations, including images, videos, and audios.



## **17. Efficient Preimage Approximation for Neural Network Certification**

cs.LG

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22798v1) [paper-pdf](http://arxiv.org/pdf/2505.22798v1)

**Authors**: Anton Björklund, Mykola Zaitsev, Marta Kwiatkowska

**Abstract**: The growing reliance on artificial intelligence in safety- and security-critical applications demands effective neural network certification. A challenging real-world use case is certification against ``patch attacks'', where adversarial patches or lighting conditions obscure parts of images, for example traffic signs. One approach to certification, which also gives quantitative coverage estimates, utilizes preimages of neural networks, i.e., the set of inputs that lead to a specified output. However, these preimage approximation methods, including the state-of-the-art PREMAP algorithm, struggle with scalability. This paper presents novel algorithmic improvements to PREMAP involving tighter bounds, adaptive Monte Carlo sampling, and improved branching heuristics. We demonstrate efficiency improvements of at least an order of magnitude on reinforcement learning control benchmarks, and show that our method scales to convolutional neural networks that were previously infeasible. Our results demonstrate the potential of preimage approximation methodology for reliability and robustness certification.



## **18. Adversarially Robust AI-Generated Image Detection for Free: An Information Theoretic Perspective**

cs.CV

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22604v1) [paper-pdf](http://arxiv.org/pdf/2505.22604v1)

**Authors**: Ruixuan Zhang, He Wang, Zhengyu Zhao, Zhiqing Guo, Xun Yang, Yunfeng Diao, Meng Wang

**Abstract**: Rapid advances in Artificial Intelligence Generated Images (AIGI) have facilitated malicious use, such as forgery and misinformation. Therefore, numerous methods have been proposed to detect fake images. Although such detectors have been proven to be universally vulnerable to adversarial attacks, defenses in this field are scarce. In this paper, we first identify that adversarial training (AT), widely regarded as the most effective defense, suffers from performance collapse in AIGI detection. Through an information-theoretic lens, we further attribute the cause of collapse to feature entanglement, which disrupts the preservation of feature-label mutual information. Instead, standard detectors show clear feature separation. Motivated by this difference, we propose Training-free Robust Detection via Information-theoretic Measures (TRIM), the first training-free adversarial defense for AIGI detection. TRIM builds on standard detectors and quantifies feature shifts using prediction entropy and KL divergence. Extensive experiments across multiple datasets and attacks validate the superiority of our TRIM, e.g., outperforming the state-of-the-art defense by 33.88% (28.91%) on ProGAN (GenImage), while well maintaining original accuracy.



## **19. AdvAgent: Controllable Blackbox Red-teaming on Web Agents**

cs.CR

ICML 2025

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2410.17401v3) [paper-pdf](http://arxiv.org/pdf/2410.17401v3)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Foundation model-based agents are increasingly used to automate complex tasks, enhancing efficiency and productivity. However, their access to sensitive resources and autonomous decision-making also introduce significant security risks, where successful attacks could lead to severe consequences. To systematically uncover these vulnerabilities, we propose AdvAgent, a black-box red-teaming framework for attacking web agents. Unlike existing approaches, AdvAgent employs a reinforcement learning-based pipeline to train an adversarial prompter model that optimizes adversarial prompts using feedback from the black-box agent. With careful attack design, these prompts effectively exploit agent weaknesses while maintaining stealthiness and controllability. Extensive evaluations demonstrate that AdvAgent achieves high success rates against state-of-the-art GPT-4-based web agents across diverse web tasks. Furthermore, we find that existing prompt-based defenses provide only limited protection, leaving agents vulnerable to our framework. These findings highlight critical vulnerabilities in current web agents and emphasize the urgent need for stronger defense mechanisms. We release code at https://ai-secure.github.io/AdvAgent/.



## **20. Understanding Adversarial Training with Energy-based Models**

cs.LG

Under review for TPAMI

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22486v1) [paper-pdf](http://arxiv.org/pdf/2505.22486v1)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Filippo Bartolucci, Senad Beadini, Giuseppe Lisanti, Iacopo Masi

**Abstract**: We aim at using Energy-based Model (EBM) framework to better understand adversarial training (AT) in classifiers, and additionally to analyze the intrinsic generative capabilities of robust classifiers. By viewing standard classifiers through an energy lens, we begin by analyzing how the energies of adversarial examples, generated by various attacks, differ from those of the natural samples. The central focus of our work is to understand the critical phenomena of Catastrophic Overfitting (CO) and Robust Overfitting (RO) in AT from an energy perspective. We analyze the impact of existing AT approaches on the energy of samples during training and observe that the behavior of the ``delta energy' -- change in energy between original sample and its adversarial counterpart -- diverges significantly when CO or RO occurs. After a thorough analysis of these energy dynamics and their relationship with overfitting, we propose a novel regularizer, the Delta Energy Regularizer (DER), designed to smoothen the energy landscape during training. We demonstrate that DER is effective in mitigating both CO and RO across multiple benchmarks. We further show that robust classifiers, when being used as generative models, have limits in handling trade-off between image quality and variability. We propose an improved technique based on a local class-wise principal component analysis (PCA) and energy-based guidance for better class-specific initialization and adaptive stopping, enhancing sample diversity and generation quality. Considering that we do not explicitly train for generative modeling, we achieve a competitive Inception Score (IS) and Fr\'echet inception distance (FID) compared to hybrid discriminative-generative models.



## **21. Adaptive Detoxification: Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing**

cs.CL

ACL 2025 Findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22298v1) [paper-pdf](http://arxiv.org/pdf/2505.22298v1)

**Authors**: Yifan Lu, Jing Li, Yigeng Zhou, Yihui Zhang, Wenya Wang, Xiucheng Li, Meishan Zhang, Fangming Liu, Jun Yu, Min Zhang

**Abstract**: Large language models (LLMs) exhibit impressive language capabilities but remain vulnerable to malicious prompts and jailbreaking attacks. Existing knowledge editing methods for LLM detoxification face two major challenges. First, they often rely on entity-specific localization, making them ineffective against adversarial inputs without explicit entities. Second, these methods suffer from over-editing, where detoxified models reject legitimate queries, compromising overall performance. In this paper, we propose ToxEdit, a toxicity-aware knowledge editing approach that dynamically detects toxic activation patterns during forward propagation. It then routes computations through adaptive inter-layer pathways to mitigate toxicity effectively. This design ensures precise toxicity mitigation while preserving LLMs' general capabilities. To more accurately assess over-editing, we also enhance the SafeEdit benchmark by incorporating instruction-following evaluation tasks. Experimental results on multiple LLMs demonstrate that our ToxEdit outperforms previous state-of-the-art methods in both detoxification performance and safeguarding general capabilities of LLMs.



## **22. Test-Time Immunization: A Universal Defense Framework Against Jailbreaks for (Multimodal) Large Language Models**

cs.CR

Under Review

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22271v1) [paper-pdf](http://arxiv.org/pdf/2505.22271v1)

**Authors**: Yongcan Yu, Yanbo Wang, Ran He, Jian Liang

**Abstract**: While (multimodal) large language models (LLMs) have attracted widespread attention due to their exceptional capabilities, they remain vulnerable to jailbreak attacks. Various defense methods are proposed to defend against jailbreak attacks, however, they are often tailored to specific types of jailbreak attacks, limiting their effectiveness against diverse adversarial strategies. For instance, rephrasing-based defenses are effective against text adversarial jailbreaks but fail to counteract image-based attacks. To overcome these limitations, we propose a universal defense framework, termed Test-time IMmunization (TIM), which can adaptively defend against various jailbreak attacks in a self-evolving way. Specifically, TIM initially trains a gist token for efficient detection, which it subsequently applies to detect jailbreak activities during inference. When jailbreak attempts are identified, TIM implements safety fine-tuning using the detected jailbreak instructions paired with refusal answers. Furthermore, to mitigate potential performance degradation in the detector caused by parameter updates during safety fine-tuning, we decouple the fine-tuning process from the detection module. Extensive experiments on both LLMs and multimodal LLMs demonstrate the efficacy of TIM.



## **23. Accountable, Scalable and DoS-resilient Secure Vehicular Communication**

cs.CR

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22162v1) [paper-pdf](http://arxiv.org/pdf/2505.22162v1)

**Authors**: Hongyu Jin, Panos Papadimitratos

**Abstract**: Paramount to vehicle safety, broadcasted Cooperative Awareness Messages (CAMs) and Decentralized Environmental Notification Messages (DENMs) are pseudonymously authenticated for security and privacy protection, with each node needing to have all incoming messages validated within an expiration deadline. This creates an asymmetry that can be easily exploited by external adversaries to launch a clogging Denial of Service (DoS) attack: each forged VC message forces all neighboring nodes to cryptographically validate it; at increasing rates, easy to generate forged messages gradually exhaust processing resources and severely degrade or deny timely validation of benign CAMs/DENMs. The result can be catastrophic when awareness of neighbor vehicle positions or critical reports are missed. We address this problem making the standardized VC pseudonymous authentication DoS-resilient. We propose efficient cryptographic constructs, which we term message verification facilitators, to prioritize processing resources for verification of potentially valid messages among bogus messages and verify multiple messages based on one signature verification. Any message acceptance is strictly based on public-key based message authentication/verification for accountability, i.e., non-repudiation is not sacrificed, unlike symmetric key based approaches. This further enables drastic misbehavior detection, also exploiting the newly introduced facilitators, based on probabilistic signature verification and cross-checking over multiple facilitators verifying the same message; while maintaining verification latency low even when under attack, trading off modest communication overhead. Our facilitators can also be used for efficient discovery and verification of DENM or any event-driven message, including misbehavior evidence used for our scheme.



## **24. The Silent Saboteur: Imperceptible Adversarial Attacks against Black-Box Retrieval-Augmented Generation Systems**

cs.IR

18 pages,accepted by ACL25 findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.18583v2) [paper-pdf](http://arxiv.org/pdf/2505.18583v2)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Jianming Lv, Maarten de Rijke, Xueqi Cheng

**Abstract**: We explore adversarial attacks against retrieval-augmented generation (RAG) systems to identify their vulnerabilities. We focus on generating human-imperceptible adversarial examples and introduce a novel imperceptible retrieve-to-generate attack against RAG. This task aims to find imperceptible perturbations that retrieve a target document, originally excluded from the initial top-$k$ candidate set, in order to influence the final answer generation. To address this task, we propose ReGENT, a reinforcement learning-based framework that tracks interactions between the attacker and the target RAG and continuously refines attack strategies based on relevance-generation-naturalness rewards. Experiments on newly constructed factual and non-factual question-answering benchmarks demonstrate that ReGENT significantly outperforms existing attack methods in misleading RAG systems with small imperceptible text perturbations.



## **25. Understanding Model Ensemble in Transferable Adversarial Attack**

cs.LG

Accepted by ICML 2025

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2410.06851v3) [paper-pdf](http://arxiv.org/pdf/2410.06851v3)

**Authors**: Wei Yao, Zeliang Zhang, Huayi Tang, Yong Liu

**Abstract**: Model ensemble adversarial attack has become a powerful method for generating transferable adversarial examples that can target even unknown models, but its theoretical foundation remains underexplored. To address this gap, we provide early theoretical insights that serve as a roadmap for advancing model ensemble adversarial attack. We first define transferability error to measure the error in adversarial transferability, alongside concepts of diversity and empirical model ensemble Rademacher complexity. We then decompose the transferability error into vulnerability, diversity, and a constant, which rigidly explains the origin of transferability error in model ensemble attack: the vulnerability of an adversarial example to ensemble components, and the diversity of ensemble components. Furthermore, we apply the latest mathematical tools in information theory to bound the transferability error using complexity and generalization terms, contributing to three practical guidelines for reducing transferability error: (1) incorporating more surrogate models, (2) increasing their diversity, and (3) reducing their complexity in cases of overfitting. Finally, extensive experiments with 54 models validate our theoretical framework, representing a significant step forward in understanding transferable model ensemble adversarial attacks.



## **26. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

cs.CR

16 pages, 12 figures, ACL 2025 findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2502.19041v2) [paper-pdf](http://arxiv.org/pdf/2502.19041v2)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.



## **27. Seeing the Threat: Vulnerabilities in Vision-Language Models to Adversarial Attack**

cs.CL

Preprint

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21967v1) [paper-pdf](http://arxiv.org/pdf/2505.21967v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Large Vision-Language Models (LVLMs) have shown remarkable capabilities across a wide range of multimodal tasks. However, their integration of visual inputs introduces expanded attack surfaces, thereby exposing them to novel security vulnerabilities. In this work, we conduct a systematic representational analysis to uncover why conventional adversarial attacks can circumvent the safety mechanisms embedded in LVLMs. We further propose a novel two stage evaluation framework for adversarial attacks on LVLMs. The first stage differentiates among instruction non compliance, outright refusal, and successful adversarial exploitation. The second stage quantifies the degree to which the model's output fulfills the harmful intent of the adversarial prompt, while categorizing refusal behavior into direct refusals, soft refusals, and partial refusals that remain inadvertently helpful. Finally, we introduce a normative schema that defines idealized model behavior when confronted with harmful prompts, offering a principled target for safety alignment in multimodal systems.



## **28. Practical Adversarial Attacks on Stochastic Bandits via Fake Data Injection**

cs.LG

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21938v1) [paper-pdf](http://arxiv.org/pdf/2505.21938v1)

**Authors**: Qirun Zeng, Eric He, Richard Hoffmann, Xuchuang Wang, Jinhang Zuo

**Abstract**: Adversarial attacks on stochastic bandits have traditionally relied on some unrealistic assumptions, such as per-round reward manipulation and unbounded perturbations, limiting their relevance to real-world systems. We propose a more practical threat model, Fake Data Injection, which reflects realistic adversarial constraints: the attacker can inject only a limited number of bounded fake feedback samples into the learner's history, simulating legitimate interactions. We design efficient attack strategies under this model, explicitly addressing both magnitude constraints (on reward values) and temporal constraints (on when and how often data can be injected). Our theoretical analysis shows that these attacks can mislead both Upper Confidence Bound (UCB) and Thompson Sampling algorithms into selecting a target arm in nearly all rounds while incurring only sublinear attack cost. Experiments on synthetic and real-world datasets validate the effectiveness of our strategies, revealing significant vulnerabilities in widely used stochastic bandit algorithms under practical adversarial scenarios.



## **29. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments**

cs.CL

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21936v1) [paper-pdf](http://arxiv.org/pdf/2505.21936v1)

**Authors**: Zeyi Liao, Jaylen Jones, Linxi Jiang, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun

**Abstract**: Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning ASRs of up to 50% in realistic end-to-end settings, with the recently released frontier Claude 4 Opus | CUA showing an alarming ASR of 48%, demonstrating that indirect prompt injection presents tangible risks for even advanced CUAs despite their capabilities and safeguards. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.



## **30. Rethinking Gradient-based Adversarial Attacks on Point Cloud Classification**

cs.CV

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21854v1) [paper-pdf](http://arxiv.org/pdf/2505.21854v1)

**Authors**: Jun Chen, Xinke Li, Mingyue Xu, Tianrui Li, Chongshou Li

**Abstract**: Gradient-based adversarial attacks have become a dominant approach for evaluating the robustness of point cloud classification models. However, existing methods often rely on uniform update rules that fail to consider the heterogeneous nature of point clouds, resulting in excessive and perceptible perturbations. In this paper, we rethink the design of gradient-based attacks by analyzing the limitations of conventional gradient update mechanisms and propose two new strategies to improve both attack effectiveness and imperceptibility. First, we introduce WAAttack, a novel framework that incorporates weighted gradients and an adaptive step-size strategy to account for the non-uniform contribution of points during optimization. This approach enables more targeted and subtle perturbations by dynamically adjusting updates according to the local structure and sensitivity of each point. Second, we propose SubAttack, a complementary strategy that decomposes the point cloud into subsets and focuses perturbation efforts on structurally critical regions. Together, these methods represent a principled rethinking of gradient-based adversarial attacks for 3D point cloud classification. Extensive experiments demonstrate that our approach outperforms state-of-the-art baselines in generating highly imperceptible adversarial examples. Code will be released upon paper acceptance.



## **31. What is Adversarial Training for Diffusion Models?**

cs.CV

40 pages

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21742v1) [paper-pdf](http://arxiv.org/pdf/2505.21742v1)

**Authors**: Briglia Maria Rosaria, Mujtaba Hussain Mirza, Giuseppe Lisanti, Iacopo Masi

**Abstract**: We answer the question in the title, showing that adversarial training (AT) for diffusion models (DMs) fundamentally differs from classifiers: while AT in classifiers enforces output invariance, AT in DMs requires equivariance to keep the diffusion process aligned with the data distribution. AT is a way to enforce smoothness in the diffusion flow, improving robustness to outliers and corrupted data. Unlike prior art, our method makes no assumptions about the noise model and integrates seamlessly into diffusion training by adding random noise, similar to randomized smoothing, or adversarial noise, akin to AT. This enables intrinsic capabilities such as handling noisy data, dealing with extreme variability such as outliers, preventing memorization, and improving robustness. We rigorously evaluate our approach with proof-of-concept datasets with known distributions in low- and high-dimensional space, thereby taking a perfect measure of errors; we further evaluate on standard benchmarks such as CIFAR-10, CelebA and LSUN Bedroom, showing strong performance under severe noise, data corruption, and iterative adversarial attacks.



## **32. Lazarus Group Targets Crypto-Wallets and Financial Data while employing new Tradecrafts**

cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21725v1) [paper-pdf](http://arxiv.org/pdf/2505.21725v1)

**Authors**: Alessio Di Santo

**Abstract**: This report presents a comprehensive analysis of a malicious software sample, detailing its architecture, behavioral characteristics, and underlying intent. Through static and dynamic examination, the malware core functionalities, including persistence mechanisms, command-and-control communication, and data exfiltration routines, are identified and its supporting infrastructure is mapped. By correlating observed indicators of compromise with known techniques, tactics, and procedures, this analysis situates the sample within the broader context of contemporary threat campaigns and infers the capabilities and motivations of its likely threat actor.   Building on these findings, actionable threat intelligence is provided to support proactive defenses. Threat hunting teams receive precise detection hypotheses for uncovering latent adversarial presence, while monitoring systems can refine alert logic to detect anomalous activity in real time. Finally, the report discusses how this structured intelligence enhances predictive risk assessments, informs vulnerability prioritization, and strengthens organizational resilience against advanced persistent threats. By integrating detailed technical insights with strategic threat landscape mapping, this malware analysis report not only reconstructs past adversary actions but also establishes a robust foundation for anticipating and mitigating future attacks.



## **33. VideoMarkBench: Benchmarking Robustness of Video Watermarking**

cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21620v1) [paper-pdf](http://arxiv.org/pdf/2505.21620v1)

**Authors**: Zhengyuan Jiang, Moyang Guo, Kecen Li, Yuepeng Hu, Yupu Wang, Zhicong Huang, Cheng Hong, Neil Zhenqiang Gong

**Abstract**: The rapid development of video generative models has led to a surge in highly realistic synthetic videos, raising ethical concerns related to disinformation and copyright infringement. Recently, video watermarking has been proposed as a mitigation strategy by embedding invisible marks into AI-generated videos to enable subsequent detection. However, the robustness of existing video watermarking methods against both common and adversarial perturbations remains underexplored. In this work, we introduce VideoMarkBench, the first systematic benchmark designed to evaluate the robustness of video watermarks under watermark removal and watermark forgery attacks. Our study encompasses a unified dataset generated by three state-of-the-art video generative models, across three video styles, incorporating four watermarking methods and seven aggregation strategies used during detection. We comprehensively evaluate 12 types of perturbations under white-box, black-box, and no-box threat models. Our findings reveal significant vulnerabilities in current watermarking approaches and highlight the urgent need for more robust solutions. Our code is available at https://github.com/zhengyuan-jiang/VideoMarkBench.



## **34. Preventing Adversarial AI Attacks Against Autonomous Situational Awareness: A Maritime Case Study**

cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21609v1) [paper-pdf](http://arxiv.org/pdf/2505.21609v1)

**Authors**: Mathew J. Walter, Aaron Barrett, Kimberly Tam

**Abstract**: Adversarial artificial intelligence (AI) attacks pose a significant threat to autonomous transportation, such as maritime vessels, that rely on AI components. Malicious actors can exploit these systems to deceive and manipulate AI-driven operations. This paper addresses three critical research challenges associated with adversarial AI: the limited scope of traditional defences, inadequate security metrics, and the need to build resilience beyond model-level defences. To address these challenges, we propose building defences utilising multiple inputs and data fusion to create defensive components and an AI security metric as a novel approach toward developing more secure AI systems. We name this approach the Data Fusion Cyber Resilience (DFCR) method, and we evaluate it through real-world demonstrations and comprehensive quantitative analyses, comparing a system built with the DFCR method against single-input models and models utilising existing state-of-the-art defences. The findings show that the DFCR approach significantly enhances resilience against adversarial machine learning attacks in maritime autonomous system operations, achieving up to a 35\% reduction in loss for successful multi-pronged perturbation attacks, up to a 100\% reduction in loss for successful adversarial patch attacks and up to 100\% reduction in loss for successful spoofing attacks when using these more resilient systems. We demonstrate how DFCR and DFCR confidence scores can reduce adversarial AI contact confidence and improve decision-making by the system, even when typical adversarial defences have been compromised. Ultimately, this work contributes to the development of more secure and resilient AI-driven systems against adversarial attacks.



## **35. AdInject: Real-World Black-Box Attacks on Web Agents via Advertising Delivery**

cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21499v1) [paper-pdf](http://arxiv.org/pdf/2505.21499v1)

**Authors**: Haowei Wang, Junjie Wang, Xiaojun Jia, Rupeng Zhang, Mingyang Li, Zhe Liu, Yang Liu, Qing Wang

**Abstract**: Vision-Language Model (VLM) based Web Agents represent a significant step towards automating complex tasks by simulating human-like interaction with websites. However, their deployment in uncontrolled web environments introduces significant security vulnerabilities. Existing research on adversarial environmental injection attacks often relies on unrealistic assumptions, such as direct HTML manipulation, knowledge of user intent, or access to agent model parameters, limiting their practical applicability. In this paper, we propose AdInject, a novel and real-world black-box attack method that leverages the internet advertising delivery to inject malicious content into the Web Agent's environment. AdInject operates under a significantly more realistic threat model than prior work, assuming a black-box agent, static malicious content constraints, and no specific knowledge of user intent. AdInject includes strategies for designing malicious ad content aimed at misleading agents into clicking, and a VLM-based ad content optimization technique that infers potential user intents from the target website's context and integrates these intents into the ad content to make it appear more relevant or critical to the agent's task, thus enhancing attack effectiveness. Experimental evaluations demonstrate the effectiveness of AdInject, attack success rates exceeding 60% in most scenarios and approaching 100% in certain cases. This strongly demonstrates that prevalent advertising delivery constitutes a potent and real-world vector for environment injection attacks against Web Agents. This work highlights a critical vulnerability in Web Agent security arising from real-world environment manipulation channels, underscoring the urgent need for developing robust defense mechanisms against such threats. Our code is available at https://github.com/NicerWang/AdInject.



## **36. Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment**

cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21494v1) [paper-pdf](http://arxiv.org/pdf/2505.21494v1)

**Authors**: Xiaojun Jia, Sensen Gao, Simeng Qin, Tianyu Pang, Chao Du, Yihao Huang, Xinfeng Li, Yiming Li, Bo Li, Yang Liu

**Abstract**: Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples. While existing methods typically achieve targeted attacks by aligning global features-such as CLIP's [CLS] token-between adversarial and target samples, they often overlook the rich local information encoded in patch tokens. This leads to suboptimal alignment and limited transferability, particularly for closed-source models. To address this limitation, we propose a targeted transferable adversarial attack method based on feature optimal alignment, called FOA-Attack, to improve adversarial transfer capability. Specifically, at the global level, we introduce a global feature loss based on cosine similarity to align the coarse-grained features of adversarial samples with those of target samples. At the local level, given the rich local representations within Transformers, we leverage clustering techniques to extract compact local patterns to alleviate redundant local features. We then formulate local feature alignment between adversarial and target samples as an optimal transport (OT) problem and propose a local clustering optimal transport loss to refine fine-grained feature alignment. Additionally, we propose a dynamic ensemble model weighting strategy to adaptively balance the influence of multiple models during adversarial example generation, thereby further improving transferability. Extensive experiments across various models demonstrate the superiority of the proposed method, outperforming state-of-the-art methods, especially in transferring to closed-source MLLMs. The code is released at https://github.com/jiaxiaojunQAQ/FOA-Attack.



## **37. When Are Concepts Erased From Diffusion Models?**

cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.17013v3) [paper-pdf](http://arxiv.org/pdf/2505.17013v3)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.



## **38. On the Robustness of Adversarial Training Against Uncertainty Attacks**

cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2410.21952v2) [paper-pdf](http://arxiv.org/pdf/2410.21952v2)

**Authors**: Emanuele Ledda, Giovanni Scodeller, Daniele Angioni, Giorgio Piras, Antonio Emanuele Cinà, Giorgio Fumera, Battista Biggio, Fabio Roli

**Abstract**: In learning problems, the noise inherent to the task at hand hinders the possibility to infer without a certain degree of uncertainty. Quantifying this uncertainty, regardless of its wide use, assumes high relevance for security-sensitive applications. Within these scenarios, it becomes fundamental to guarantee good (i.e., trustworthy) uncertainty measures, which downstream modules can securely employ to drive the final decision-making process. However, an attacker may be interested in forcing the system to produce either (i) highly uncertain outputs jeopardizing the system's availability or (ii) low uncertainty estimates, making the system accept uncertain samples that would instead require a careful inspection (e.g., human intervention). Therefore, it becomes fundamental to understand how to obtain robust uncertainty estimates against these kinds of attacks. In this work, we reveal both empirically and theoretically that defending against adversarial examples, i.e., carefully perturbed samples that cause misclassification, additionally guarantees a more secure, trustworthy uncertainty estimate under common attack scenarios without the need for an ad-hoc defense strategy. To support our claims, we evaluate multiple adversarial-robust models from the publicly available benchmark RobustBench on the CIFAR-10 and ImageNet datasets.



## **39. Attribute-Efficient PAC Learning of Sparse Halfspaces with Constant Malicious Noise Rate**

cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21430v1) [paper-pdf](http://arxiv.org/pdf/2505.21430v1)

**Authors**: Shiwei Zeng, Jie Shen

**Abstract**: Attribute-efficient learning of sparse halfspaces has been a fundamental problem in machine learning theory. In recent years, machine learning algorithms are faced with prevalent data corruptions or even adversarial attacks. It is of central interest to design efficient algorithms that are robust to noise corruptions. In this paper, we consider that there exists a constant amount of malicious noise in the data and the goal is to learn an underlying $s$-sparse halfspace $w^* \in \mathbb{R}^d$ with $\text{poly}(s,\log d)$ samples. Specifically, we follow a recent line of works and assume that the underlying distribution satisfies a certain concentration condition and a margin condition at the same time. Under such conditions, we show that attribute-efficiency can be achieved by simple variants to existing hinge loss minimization programs. Our key contribution includes: 1) an attribute-efficient PAC learning algorithm that works under constant malicious noise rate; 2) a new gradient analysis that carefully handles the sparsity constraint in hinge loss minimization.



## **40. A Framework for Adversarial Analysis of Decision Support Systems Prior to Deployment**

cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21414v1) [paper-pdf](http://arxiv.org/pdf/2505.21414v1)

**Authors**: Brett Bissey, Kyle Gatesman, Walker Dimon, Mohammad Alam, Luis Robaina, Joseph Weissman

**Abstract**: This paper introduces a comprehensive framework designed to analyze and secure decision-support systems trained with Deep Reinforcement Learning (DRL), prior to deployment, by providing insights into learned behavior patterns and vulnerabilities discovered through simulation. The introduced framework aids in the development of precisely timed and targeted observation perturbations, enabling researchers to assess adversarial attack outcomes within a strategic decision-making context. We validate our framework, visualize agent behavior, and evaluate adversarial outcomes within the context of a custom-built strategic game, CyberStrike. Utilizing the proposed framework, we introduce a method for systematically discovering and ranking the impact of attacks on various observation indices and time-steps, and we conduct experiments to evaluate the transferability of adversarial attacks across agent architectures and DRL training algorithms. The findings underscore the critical need for robust adversarial defense mechanisms to protect decision-making policies in high-stakes environments.



## **41. Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model Approach**

cs.LG

15 pages, 7 figures, accepted by ICML 2025

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2502.06832v3) [paper-pdf](http://arxiv.org/pdf/2502.06832v3)

**Authors**: Xu Zhang, Kaidi Xu, Ziqing Hu, Ren Wang

**Abstract**: Mixture of Experts (MoE) have shown remarkable success in leveraging specialized expert networks for complex machine learning tasks. However, their susceptibility to adversarial attacks presents a critical challenge for deployment in robust applications. This paper addresses the critical question of how to incorporate robustness into MoEs while maintaining high natural accuracy. We begin by analyzing the vulnerability of MoE components, finding that expert networks are notably more susceptible to adversarial attacks than the router. Based on this insight, we propose a targeted robust training technique that integrates a novel loss function to enhance the adversarial robustness of MoE, requiring only the robustification of one additional expert without compromising training or inference efficiency. Building on this, we introduce a dual-model strategy that linearly combines a standard MoE model with our robustified MoE model using a smoothing parameter. This approach allows for flexible control over the robustness-accuracy trade-off. We further provide theoretical foundations by deriving certified robustness bounds for both the single MoE and the dual-model. To push the boundaries of robustness and accuracy, we propose a novel joint training strategy JTDMoE for the dual-model. This joint training enhances both robustness and accuracy beyond what is achievable with separate models. Experimental results on CIFAR-10 and TinyImageNet datasets using ResNet18 and Vision Transformer (ViT) architectures demonstrate the effectiveness of our proposed methods. The code is publicly available at https://github.com/TIML-Group/Robust-MoE-Dual-Model.



## **42. Boosting Adversarial Transferability via High-Frequency Augmentation and Hierarchical-Gradient Fusion**

cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21181v1) [paper-pdf](http://arxiv.org/pdf/2505.21181v1)

**Authors**: Yayin Zheng, Chen Wan, Zihong Guo, Hailing Kuang, Xiaohai Lu

**Abstract**: Adversarial attacks have become a significant challenge in the security of machine learning models, particularly in the context of black-box defense strategies. Existing methods for enhancing adversarial transferability primarily focus on the spatial domain. This paper presents Frequency-Space Attack (FSA), a new adversarial attack framework that effectively integrates frequency-domain and spatial-domain transformations. FSA combines two key techniques: (1) High-Frequency Augmentation, which applies Fourier transform with frequency-selective amplification to diversify inputs and emphasize the critical role of high-frequency components in adversarial attacks, and (2) Hierarchical-Gradient Fusion, which merges multi-scale gradient decomposition and fusion to capture both global structures and fine-grained details, resulting in smoother perturbations. Our experiment demonstrates that FSA consistently outperforms state-of-the-art methods across various black-box models. Notably, our proposed FSA achieves an average attack success rate increase of 23.6% compared with BSR (CVPR 2024) on eight black-box defense models.



## **43. Unraveling Indirect In-Context Learning Using Influence Functions**

cs.LG

Under Review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2501.01473v2) [paper-pdf](http://arxiv.org/pdf/2501.01473v2)

**Authors**: Hadi Askari, Shivanshu Gupta, Terry Tong, Fei Wang, Anshuman Chhabra, Muhao Chen

**Abstract**: In this work, we introduce a novel paradigm for generalized In-Context Learning (ICL), termed Indirect In-Context Learning. In Indirect ICL, we explore demonstration selection strategies tailored for two distinct real-world scenarios: Mixture of Tasks and Noisy ICL. We systematically evaluate the effectiveness of Influence Functions (IFs) as a selection tool for these settings, highlighting the potential of IFs to better capture the informativeness of examples within the demonstration pool. For the Mixture of Tasks setting, demonstrations are drawn from 28 diverse tasks, including MMLU, BigBench, StrategyQA, and CommonsenseQA. We demonstrate that combining BertScore-Recall (BSR) with an IF surrogate model can further improve performance, leading to average absolute accuracy gains of 0.37\% and 1.45\% for 3-shot and 5-shot setups when compared to traditional ICL metrics. In the Noisy ICL setting, we examine scenarios where demonstrations might be mislabeled or have adversarial noise. Our experiments show that reweighting traditional ICL selectors (BSR and Cosine Similarity) with IF-based selectors boosts accuracy by an average of 2.90\% for Cosine Similarity and 2.94\% for BSR on noisy GLUE benchmarks. For the adversarial sub-setting, we show the utility of using IFs for task-agnostic demonstration selection for backdoor attack mitigation. Showing a 32.89\% reduction in Attack Success Rate compared to task-aware methods. In sum, we propose a robust framework for demonstration selection that generalizes beyond traditional ICL, offering valuable insights into the role of IFs for Indirect ICL.



## **44. TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data**

cs.LG

63 pages, 22 figures, 6 tables

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21027v1) [paper-pdf](http://arxiv.org/pdf/2505.21027v1)

**Authors**: Zhipeng He, Chun Ouyang, Lijie Wen, Cong Liu, Catarina Moreira

**Abstract**: Adversarial attacks pose a significant threat to machine learning models by inducing incorrect predictions through imperceptible perturbations to input data. While these attacks have been extensively studied in unstructured data like images, their application to tabular data presents new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ significantly from those in image data. To address these differences, it is crucial to consider imperceptibility as a key criterion specific to tabular data. Most current research focuses primarily on achieving effective adversarial attacks, often overlooking the importance of maintaining imperceptibility. To address this gap, we propose a new benchmark for adversarial attacks on tabular data that evaluates both effectiveness and imperceptibility. In this study, we assess the effectiveness and imperceptibility of five adversarial attacks across four models using eleven tabular datasets, including both mixed and numerical-only datasets. Our analysis explores how these factors interact and influence the overall performance of the attacks. We also compare the results across different dataset types to understand the broader implications of these findings. The findings from this benchmark provide valuable insights for improving the design of adversarial attack algorithms, thereby advancing the field of adversarial machine learning on tabular data.



## **45. Tradeoffs Between Alignment and Helpfulness in Language Models with Steering Methods**

cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2401.16332v5) [paper-pdf](http://arxiv.org/pdf/2401.16332v5)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.



## **46. NatADiff: Adversarial Boundary Guidance for Natural Adversarial Diffusion**

cs.LG

10 pages, 3 figures, 2 tables

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20934v1) [paper-pdf](http://arxiv.org/pdf/2505.20934v1)

**Authors**: Max Collins, Jordan Vice, Tim French, Ajmal Mian

**Abstract**: Adversarial samples exploit irregularities in the manifold ``learned'' by deep learning models to cause misclassifications. The study of these adversarial samples provides insight into the features a model uses to classify inputs, which can be leveraged to improve robustness against future attacks. However, much of the existing literature focuses on constrained adversarial samples, which do not accurately reflect test-time errors encountered in real-world settings. To address this, we propose `NatADiff', an adversarial sampling scheme that leverages denoising diffusion to generate natural adversarial samples. Our approach is based on the observation that natural adversarial samples frequently contain structural elements from the adversarial class. Deep learning models can exploit these structural elements to shortcut the classification process, rather than learning to genuinely distinguish between classes. To leverage this behavior, we guide the diffusion trajectory towards the intersection of the true and adversarial classes, combining time-travel sampling with augmented classifier guidance to enhance attack transferability while preserving image fidelity. Our method achieves comparable attack success rates to current state-of-the-art techniques, while exhibiting significantly higher transferability across model architectures and better alignment with natural test-time errors as measured by FID. These results demonstrate that NatADiff produces adversarial samples that not only transfer more effectively across models, but more faithfully resemble naturally occurring test-time errors.



## **47. ZeroPur: Succinct Training-Free Adversarial Purification**

cs.CV

17 pages, 7 figures, under review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2406.03143v3) [paper-pdf](http://arxiv.org/pdf/2406.03143v3)

**Authors**: Erhu Liu, Zonglin Yang, Bo Liu, Bin Xiao, Xiuli Bi

**Abstract**: Adversarial purification is a kind of defense technique that can defend against various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold, and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.



## **48. Concealment of Intent: A Game-Theoretic Analysis**

cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20841v1) [paper-pdf](http://arxiv.org/pdf/2505.20841v1)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.



## **49. MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20824v1) [paper-pdf](http://arxiv.org/pdf/2505.20824v1)

**Authors**: Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, Yang Gao

**Abstract**: As large language models (LLMs) are increasingly deployed in healthcare, ensuring their safety, particularly within collaborative multi-agent configurations, is paramount. In this paper we introduce MedSentry, a benchmark comprising 5 000 adversarial medical prompts spanning 25 threat categories with 100 subthemes. Coupled with this dataset, we develop an end-to-end attack-defense evaluation pipeline to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) withstand attacks from 'dark-personality' agents. Our findings reveal critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For instance, SharedPool's open information sharing makes it highly susceptible, whereas Decentralized architectures exhibit greater resilience thanks to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring system safety to near-baseline levels. MedSentry thus furnishes both a rigorous evaluation framework and practical defense strategies that guide the design of safer LLM-based multi-agent systems in medical domains.



## **50. TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent**

cs.CL

9 pages, 5 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20118v2) [paper-pdf](http://arxiv.org/pdf/2505.20118v2)

**Authors**: Dominik Meier, Jan Philip Wahle, Paul Röttger, Terry Ruas, Bela Gipp

**Abstract**: As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.



