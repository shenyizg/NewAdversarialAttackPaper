# Latest Adversarial Attack Papers
**update at 2025-04-07 09:23:55**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. SoK: Attacks on Modern Card Payments**

cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03363v1) [paper-pdf](http://arxiv.org/pdf/2504.03363v1)

**Authors**: Xenia Hofmeier, David Basin, Ralf Sasse, Jorge Toro-Pozo

**Abstract**: EMV is the global standard for smart card payments and its NFC-based version, EMV contactless, is widely used, also for mobile payments. In this systematization of knowledge, we examine attacks on the EMV contactless protocol. We provide a comprehensive framework encompassing its desired security properties and adversary models. We also identify and categorize a comprehensive collection of protocol flaws and show how different subsets thereof can be combined into attacks. In addition to this systematization, we examine the underlying reasons for the many attacks against EMV and point to a better way forward.



## **2. SLACK: Attacking LiDAR-based SLAM with Adversarial Point Injections**

cs.CV

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03089v1) [paper-pdf](http://arxiv.org/pdf/2504.03089v1)

**Authors**: Prashant Kumar, Dheeraj Vattikonda, Kshitij Madhav Bhat, Kunal Dargan, Prem Kalra

**Abstract**: The widespread adoption of learning-based methods for the LiDAR makes autonomous vehicles vulnerable to adversarial attacks through adversarial \textit{point injections (PiJ)}. It poses serious security challenges for navigation and map generation. Despite its critical nature, no major work exists that studies learning-based attacks on LiDAR-based SLAM. Our work proposes SLACK, an end-to-end deep generative adversarial model to attack LiDAR scans with several point injections without deteriorating LiDAR quality. To facilitate SLACK, we design a novel yet simple autoencoder that augments contrastive learning with segmentation-based attention for precise reconstructions. SLACK demonstrates superior performance on the task of \textit{point injections (PiJ)} compared to the best baselines on KITTI and CARLA-64 dataset while maintaining accurate scan quality. We qualitatively and quantitatively demonstrate PiJ attacks using a fraction of LiDAR points. It severely degrades navigation and map quality without deteriorating the LiDAR scan quality.



## **3. Integrating Identity-Based Identification against Adaptive Adversaries in Federated Learning**

cs.CR

10 pages, 5 figures, research article, IEEE possible publication (in  submission)

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03077v1) [paper-pdf](http://arxiv.org/pdf/2504.03077v1)

**Authors**: Jakub Kacper Szelag, Ji-Jian Chin, Lauren Ansell, Sook-Chin Yip

**Abstract**: Federated Learning (FL) has recently emerged as a promising paradigm for privacy-preserving, distributed machine learning. However, FL systems face significant security threats, particularly from adaptive adversaries capable of modifying their attack strategies to evade detection. One such threat is the presence of Reconnecting Malicious Clients (RMCs), which exploit FLs open connectivity by reconnecting to the system with modified attack strategies. To address this vulnerability, we propose integration of Identity-Based Identification (IBI) as a security measure within FL environments. By leveraging IBI, we enable FL systems to authenticate clients based on cryptographic identity schemes, effectively preventing previously disconnected malicious clients from re-entering the system. Our approach is implemented using the TNC-IBI (Tan-Ng-Chin) scheme over elliptic curves to ensure computational efficiency, particularly in resource-constrained environments like Internet of Things (IoT). Experimental results demonstrate that integrating IBI with secure aggregation algorithms, such as Krum and Trimmed Mean, significantly improves FL robustness by mitigating the impact of RMCs. We further discuss the broader implications of IBI in FL security, highlighting research directions for adaptive adversary detection, reputation-based mechanisms, and the applicability of identity-based cryptographic frameworks in decentralized FL architectures. Our findings advocate for a holistic approach to FL security, emphasizing the necessity of proactive defence strategies against evolving adaptive adversarial threats.



## **4. Moving Target Defense Against Adversarial False Data Injection Attacks In Power Grids**

eess.SY

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03065v1) [paper-pdf](http://arxiv.org/pdf/2504.03065v1)

**Authors**: Yexiang Chen, Subhash Lakshminarayana, H. Vincent Poor

**Abstract**: Machine learning (ML)-based detectors have been shown to be effective in detecting stealthy false data injection attacks (FDIAs) that can bypass conventional bad data detectors (BDDs) in power systems. However, ML models are also vulnerable to adversarial attacks. A sophisticated perturbation signal added to the original BDD-bypassing FDIA can conceal the attack from ML-based detectors. In this paper, we develop a moving target defense (MTD) strategy to defend against adversarial FDIAs in power grids. We first develop an MTD-strengthened deep neural network (DNN) model, which deploys a pool of DNN models rather than a single static model that cooperate to detect the adversarial attack jointly. The MTD model pool introduces randomness to the ML model's decision boundary, thereby making the adversarial attacks detectable. Furthermore, to increase the effectiveness of the MTD strategy and reduce the computational costs associated with developing the MTD model pool, we combine this approach with the physics-based MTD, which involves dynamically perturbing the transmission line reactance and retraining the DNN-based detector to adapt to the new system topology. Simulations conducted on IEEE test bus systems demonstrate that the MTD-strengthened DNN achieves up to 94.2% accuracy in detecting adversarial FDIAs. When combined with a physics-based MTD, the detection accuracy surpasses 99%, while significantly reducing the computational costs of updating the DNN models. This approach requires only moderate perturbations to transmission line reactances, resulting in minimal increases in OPF cost.



## **5. Federated Learning in Adversarial Environments: Testbed Design and Poisoning Resilience in Cybersecurity**

cs.CR

6 pages, 4 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2409.09794v2) [paper-pdf](http://arxiv.org/pdf/2409.09794v2)

**Authors**: Hao Jian Huang, Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: This paper presents the design and implementation of a Federated Learning (FL) testbed, focusing on its application in cybersecurity and evaluating its resilience against poisoning attacks. Federated Learning allows multiple clients to collaboratively train a global model while keeping their data decentralized, addressing critical needs for data privacy and security, particularly in sensitive fields like cybersecurity. Our testbed, built using Raspberry Pi and Nvidia Jetson hardware by running the Flower framework, facilitates experimentation with various FL frameworks, assessing their performance, scalability, and ease of integration. Through a case study on federated intrusion detection systems, the testbed's capabilities are shown in detecting anomalies and securing critical infrastructure without exposing sensitive network data. Comprehensive poisoning tests, targeting both model and data integrity, evaluate the system's robustness under adversarial conditions. The results show that while federated learning enhances data privacy and distributed learning, it remains vulnerable to poisoning attacks, which must be mitigated to ensure its reliability in real-world applications.



## **6. ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization**

cs.CL

18 pages, 5 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02725v1) [paper-pdf](http://arxiv.org/pdf/2504.02725v1)

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency.



## **7. No Free Lunch with Guardrails**

cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.00441v2) [paper-pdf](http://arxiv.org/pdf/2504.00441v2)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.   Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and Enkrypt AI guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability.



## **8. A Survey and Evaluation of Adversarial Attacks for Object Detection**

cs.CV

17 pages

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2408.01934v3) [paper-pdf](http://arxiv.org/pdf/2408.01934v3)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models achieve remarkable accuracy in computer vision tasks, yet remain vulnerable to adversarial examples--carefully crafted perturbations to input images that can deceive these models into making confident but incorrect predictions. This vulnerability pose significant risks in high-stakes applications such as autonomous vehicles, security surveillance, and safety-critical inspection systems. While the existing literature extensively covers adversarial attacks in image classification, comprehensive analyses of such attacks on object detection systems remain limited. This paper presents a novel taxonomic framework for categorizing adversarial attacks specific to object detection architectures, synthesizes existing robustness metrics, and provides a comprehensive empirical evaluation of state-of-the-art attack methodologies on popular object detection models, including both traditional detectors and modern detectors with vision-language pretraining. Through rigorous analysis of open-source attack implementations and their effectiveness across diverse detection architectures, we derive key insights into attack characteristics. Furthermore, we delineate critical research gaps and emerging challenges to guide future investigations in securing object detection systems against adversarial threats. Our findings establish a foundation for developing more robust detection models while highlighting the urgent need for standardized evaluation protocols in this rapidly evolving domain.



## **9. Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems**

cs.LG

accepted by CVPR2025

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2503.00383v2) [paper-pdf](http://arxiv.org/pdf/2503.00383v2)

**Authors**: Song Xia, Yi Yu, Wenhan Yang, Meiwen Ding, Zhuo Chen, Ling-Yu Duan, Alex C. Kot, Xudong Jiang

**Abstract**: By locally encoding raw data into intermediate features, collaborative inference enables end users to leverage powerful deep learning models without exposure of sensitive raw data to cloud servers. However, recent studies have revealed that these intermediate features may not sufficiently preserve privacy, as information can be leaked and raw data can be reconstructed via model inversion attacks (MIAs). Obfuscation-based methods, such as noise corruption, adversarial representation learning, and information filters, enhance the inversion robustness by obfuscating the task-irrelevant redundancy empirically. However, methods for quantifying such redundancy remain elusive, and the explicit mathematical relation between this redundancy minimization and inversion robustness enhancement has not yet been established. To address that, this work first theoretically proves that the conditional entropy of inputs given intermediate features provides a guaranteed lower bound on the reconstruction mean square error (MSE) under any MIA. Then, we derive a differentiable and solvable measure for bounding this conditional entropy based on the Gaussian mixture estimation and propose a conditional entropy maximization (CEM) algorithm to enhance the inversion robustness. Experimental results on four datasets demonstrate the effectiveness and adaptability of our proposed CEM; without compromising feature utility and computing efficiency, plugging the proposed CEM into obfuscation-based defense mechanisms consistently boosts their inversion robustness, achieving average gains ranging from 12.9\% to 48.2\%. Code is available at \href{https://github.com/xiasong0501/CEM}{https://github.com/xiasong0501/CEM}.



## **10. Robust Unsupervised Domain Adaptation for 3D Point Cloud Segmentation Under Source Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.01659v2) [paper-pdf](http://arxiv.org/pdf/2504.01659v2)

**Authors**: Haosheng Li, Junjie Chen, Yuecong Xu, Kemi Ding

**Abstract**: Unsupervised domain adaptation (UDA) frameworks have shown good generalization capabilities for 3D point cloud semantic segmentation models on clean data. However, existing works overlook adversarial robustness when the source domain itself is compromised. To comprehensively explore the robustness of the UDA frameworks, we first design a stealthy adversarial point cloud generation attack that can significantly contaminate datasets with only minor perturbations to the point cloud surface. Based on that, we propose a novel dataset, AdvSynLiDAR, comprising synthesized contaminated LiDAR point clouds. With the generated corrupted data, we further develop the Adversarial Adaptation Framework (AAF) as the countermeasure. Specifically, by extending the key point sensitive (KPS) loss towards the Robust Long-Tail loss (RLT loss) and utilizing a decoder branch, our approach enables the model to focus on long-tail classes during the pre-training phase and leverages high-confidence decoded point cloud information to restore point cloud structures during the adaptation phase. We evaluated our AAF method on the AdvSynLiDAR dataset, where the results demonstrate that our AAF method can mitigate performance degradation under source adversarial perturbations for UDA in the 3D point cloud segmentation application.



## **11. Secure Generalization through Stochastic Bidirectional Parameter Updates Using Dual-Gradient Mechanism**

cs.LG

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02213v1) [paper-pdf](http://arxiv.org/pdf/2504.02213v1)

**Authors**: Shourya Goel, Himanshi Tibrewal, Anant Jain, Anshul Pundhir, Pravendra Singh

**Abstract**: Federated learning (FL) has gained increasing attention due to privacy-preserving collaborative training on decentralized clients, mitigating the need to upload sensitive data to a central server directly. Nonetheless, recent research has underscored the risk of exposing private data to adversaries, even within FL frameworks. In general, existing methods sacrifice performance while ensuring resistance to privacy leakage in FL. We overcome these issues and generate diverse models at a global server through the proposed stochastic bidirectional parameter update mechanism. Using diverse models, we improved the generalization and feature representation in the FL setup, which also helped to improve the robustness of the model against privacy leakage without hurting the model's utility. We use global models from past FL rounds to follow systematic perturbation in parameter space at the server to ensure model generalization and resistance against privacy attacks. We generate diverse models (in close neighborhoods) for each client by using systematic perturbations in model parameters at a fine-grained level (i.e., altering each convolutional filter across the layers of the model) to improve the generalization and security perspective. We evaluated our proposed approach on four benchmark datasets to validate its superiority. We surpassed the state-of-the-art methods in terms of model utility and robustness towards privacy leakage. We have proven the effectiveness of our method by evaluating performance using several quantitative and qualitative results.



## **12. FairDAG: Consensus Fairness over Concurrent Causal Design**

cs.DB

17 pages, 15 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02194v1) [paper-pdf](http://arxiv.org/pdf/2504.02194v1)

**Authors**: Dakai Kang, Junchao Chen, Tien Tuan Anh Dinh, Mohammad Sadoghi

**Abstract**: The rise of cryptocurrencies like Bitcoin and Ethereum has driven interest in blockchain technology, with Ethereum's smart contracts enabling the growth of decentralized finance (DeFi). However, research has shown that adversaries exploit transaction ordering to extract profits through attacks like front-running, sandwich attacks, and liquidation manipulation. This issue affects both permissionless and permissioned blockchains, as block proposers have full control over transaction ordering. To address this, a more fair approach to transaction ordering is essential.   Existing fairness protocols, such as Pompe and Themis, operate on leader-based consensus protocols, which not only suffer from low throughput but also allow adversaries to manipulate transaction ordering. To address these limitations, we propose FairDAG-AB and FairDAG-RL, which leverage DAG-based consensus protocols.   We theoretically demonstrate that FairDAG protocols not only uphold fairness guarantees, as previous fairness protocols do, but also achieve higher throughput and greater resilience to adversarial ordering manipulation. Our deployment and evaluation on CloudLab further validate these claims.



## **13. Learning to Lie: Reinforcement Learning Attacks Damage Human-AI Teams and Teams of LLMs**

cs.HC

17 pages, 9 figures, accepted to ICLR 2025 Workshop on Human-AI  Coevolution

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2503.21983v2) [paper-pdf](http://arxiv.org/pdf/2503.21983v2)

**Authors**: Abed Kareem Musaffar, Anand Gokhale, Sirui Zeng, Rasta Tadayon, Xifeng Yan, Ambuj Singh, Francesco Bullo

**Abstract**: As artificial intelligence (AI) assistants become more widely adopted in safety-critical domains, it becomes important to develop safeguards against potential failures or adversarial attacks. A key prerequisite to developing these safeguards is understanding the ability of these AI assistants to mislead human teammates. We investigate this attack problem within the context of an intellective strategy game where a team of three humans and one AI assistant collaborate to answer a series of trivia questions. Unbeknownst to the humans, the AI assistant is adversarial. Leveraging techniques from Model-Based Reinforcement Learning (MBRL), the AI assistant learns a model of the humans' trust evolution and uses that model to manipulate the group decision-making process to harm the team. We evaluate two models -- one inspired by literature and the other data-driven -- and find that both can effectively harm the human team. Moreover, we find that in this setting our data-driven model is capable of accurately predicting how human agents appraise their teammates given limited information on prior interactions. Finally, we compare the performance of state-of-the-art LLM models to human agents on our influence allocation task to evaluate whether the LLMs allocate influence similarly to humans or if they are more robust to our attack. These results enhance our understanding of decision-making dynamics in small human-AI teams and lay the foundation for defense strategies.



## **14. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

Included in Proceedings of the Conference on Applied Machine Learning  in Information Security (CAMLIS 2024), Arlington, Virginia, USA, October  24-25, 2024

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2406.03230v5) [paper-pdf](http://arxiv.org/pdf/2406.03230v5)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **15. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

cs.CL

8 pages, 6 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02132v1) [paper-pdf](http://arxiv.org/pdf/2504.02132v1)

**Authors**: Ezzeldin Shereen, Dan Ristea, Burak Hasircioglu, Shae McFadden, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multimodal retrieval augmented generation (M-RAG) has recently emerged as a method to inhibit hallucinations of large multimodal models (LMMs) through a factual knowledge base (KB). However, M-RAG also introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this work, we present a poisoning attack against M-RAG targeting visual document retrieval applications, where the KB contains images of document pages. Our objective is to craft a single image that is retrieved for a variety of different user queries, and consistently influences the output produced by the generative model, thus creating a universal denial-of-service (DoS) attack against the M-RAG system. We demonstrate that while our attack is effective against a diverse range of widely-used, state-of-the-art retrievers (embedding models) and generators (LMMs), it can also be ineffective against robust embedding models. Our attack not only highlights the vulnerability of M-RAG pipelines to poisoning attacks, but also sheds light on a fundamental weakness that potentially hinders their performance even in benign settings.



## **16. Graph Analytics for Cyber-Physical System Resilience Quantification**

cs.CR

32 pages, 11 figures, 3 tables

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02120v1) [paper-pdf](http://arxiv.org/pdf/2504.02120v1)

**Authors**: Romain Dagnas, Michel Barbeau, Joaquin Garcia-Alfaro, Reda Yaich

**Abstract**: Critical infrastructures integrate a wide range of smart technologies and become highly connected to the cyber world. This is especially true for Cyber-Physical Systems (CPSs), which integrate hardware and software components. Despite the advantages of smart infrastructures, they remain vulnerable to cyberattacks. This work focuses on the cyber resilience of CPSs. We propose a methodology based on knowledge graph modeling and graph analytics to quantify the resilience potential of complex systems by using a multilayered model based on knowledge graphs. Our methodology also allows us to identify critical points. These critical points are components or functions of an architecture that can generate critical failures if attacked. Thus, identifying them can help enhance resilience and avoid cascading effects. We use the SWaT (Secure Water Treatment) testbed as a use case to achieve this objective. This system mimics the actual behavior of a water treatment station in Singapore. We model three resilient designs of SWaT according to our multilayered model. We conduct a resilience assessment based on three relevant metrics used in graph analytics. We compare the results obtained with each metric and discuss their accuracy in identifying critical points. We perform an experimentation analysis based on the knowledge gained by a cyber adversary about the system architecture. We show that the most resilient SWaT design has the necessary potential to bounce back and absorb the attacks. We discuss our results and conclude this work by providing further research axes.



## **17. AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization**

cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01735v1) [paper-pdf](http://arxiv.org/pdf/2504.01735v1)

**Authors**: Chaohu Liu, Tianyi Gui, Yu Liu, Linli Xu

**Abstract**: Large Vision-Language Models (LVLMs), such as GPT-4o and LLaVA, have recently witnessed remarkable advancements and are increasingly being deployed in real-world applications. However, inheriting the sensitivity of visual neural networks, LVLMs remain vulnerable to adversarial attacks, which can result in erroneous or malicious outputs. While existing efforts utilize adversarial fine-tuning to enhance robustness, they often suffer from performance degradation on clean inputs. In this paper, we proposes AdPO, a novel adversarial defense strategy for LVLMs based on preference optimization. For the first time, we reframe adversarial training as a preference optimization problem, aiming to enhance the model's preference for generating normal outputs on clean inputs while rejecting the potential misleading outputs for adversarial examples. Notably, AdPO achieves this by solely modifying the image encoder, e.g., CLIP ViT, resulting in superior clean and adversarial performance in a variety of downsream tasks. Considering that training involves large language models (LLMs), the computational cost increases significantly. We validate that training on smaller LVLMs and subsequently transferring to larger models can achieve competitive performance while maintaining efficiency comparable to baseline methods. Our comprehensive experiments confirm the effectiveness of the proposed AdPO, which provides a novel perspective for future adversarial defense research.



## **18. Sky of Unlearning (SoUL): Rewiring Federated Machine Unlearning via Selective Pruning**

cs.LG

6 pages, 6 figures, IEEE International Conference on Communications  (ICC 2025)

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01705v1) [paper-pdf](http://arxiv.org/pdf/2504.01705v1)

**Authors**: Md Mahabub Uz Zaman, Xiang Sun, Jingjing Yao

**Abstract**: The Internet of Drones (IoD), where drones collaborate in data collection and analysis, has become essential for applications such as surveillance and environmental monitoring. Federated learning (FL) enables drones to train machine learning models in a decentralized manner while preserving data privacy. However, FL in IoD networks is susceptible to attacks like data poisoning and model inversion. Federated unlearning (FU) mitigates these risks by eliminating adversarial data contributions, preventing their influence on the model. This paper proposes sky of unlearning (SoUL), a federated unlearning framework that efficiently removes the influence of unlearned data while maintaining model performance. A selective pruning algorithm is designed to identify and remove neurons influential in unlearning but minimally impact the overall performance of the model. Simulations demonstrate that SoUL outperforms existing unlearning methods, achieves accuracy comparable to full retraining, and reduces computation and communication overhead, making it a scalable and efficient solution for resource-constrained IoD networks.



## **19. Overlap-Aware Feature Learning for Robust Unsupervised Domain Adaptation for 3D Semantic Segmentation**

cs.CV

8 pages,6 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01668v1) [paper-pdf](http://arxiv.org/pdf/2504.01668v1)

**Authors**: Junjie Chen, Yuecong Xu, Haosheng Li, Kemi Ding

**Abstract**: 3D point cloud semantic segmentation (PCSS) is a cornerstone for environmental perception in robotic systems and autonomous driving, enabling precise scene understanding through point-wise classification. While unsupervised domain adaptation (UDA) mitigates label scarcity in PCSS, existing methods critically overlook the inherent vulnerability to real-world perturbations (e.g., snow, fog, rain) and adversarial distortions. This work first identifies two intrinsic limitations that undermine current PCSS-UDA robustness: (a) unsupervised features overlap from unaligned boundaries in shared-class regions and (b) feature structure erosion caused by domain-invariant learning that suppresses target-specific patterns. To address the proposed problems, we propose a tripartite framework consisting of: 1) a robustness evaluation model quantifying resilience against adversarial attack/corruption types through robustness metrics; 2) an invertible attention alignment module (IAAM) enabling bidirectional domain mapping while preserving discriminative structure via attention-guided overlap suppression; and 3) a contrastive memory bank with quality-aware contrastive learning that progressively refines pseudo-labels with feature quality for more discriminative representations. Extensive experiments on SynLiDAR-to-SemanticPOSS adaptation demonstrate a maximum mIoU improvement of 14.3\% under adversarial attack.



## **20. Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions**

cs.CV

Under review

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01632v1) [paper-pdf](http://arxiv.org/pdf/2504.01632v1)

**Authors**: Giulia Marchiori Pietrosanti, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: The robustness of DNNs is a crucial factor in safety-critical applications, particularly in complex and dynamic environments where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remained underexplored. This paper fills this gap by introducing specialized metrics for benchmarking the spatial robustness of segmentation models, alongside with an evaluation framework to assess the impact of localized corruptions. Furthermore, we uncover the inherent complexity of characterizing worst-case robustness using a single localized adversarial perturbation. To address this, we propose region-aware multi-attack adversarial analysis, a method that enables a deeper understanding of model robustness against adversarial perturbations applied to specific regions. The proposed metrics and analysis were evaluated on 15 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones and vice-versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks.



## **21. Representation Bending for Large Language Model Safety**

cs.LG

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01550v1) [paper-pdf](http://arxiv.org/pdf/2504.01550v1)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **22. A Volumetric Approach to Privacy of Dynamical Systems**

eess.SY

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2501.02893v3) [paper-pdf](http://arxiv.org/pdf/2501.02893v3)

**Authors**: Chuanghong Weng, Ehsan Nekouei

**Abstract**: Information-theoretic metrics, such as mutual information, have been widely used to evaluate privacy leakage in dynamic systems. However, these approaches are typically limited to stochastic systems and face computational challenges. In this paper, we introduce a novel volumetric framework for analyzing privacy in systems affected by unknown but bounded noise. Our model considers a dynamic system comprising public and private states, where an observation set of the public state is released. An adversary utilizes the observed public state to infer an uncertainty set of the private state, referred to as the inference attack. We define the evolution dynamics of these inference attacks and quantify the privacy level of the private state using the volume of its uncertainty sets. We then develop an approximate computation method leveraging interval analysis to compute the private state set. We investigate the properties of the proposed volumetric privacy measure and demonstrate that it is bounded by the information gain derived from the observation set. Furthermore, we propose an optimization approach to designing privacy filter using randomization and linear programming based on the proposed privacy measure. The effectiveness of the optimal privacy filter design is evaluated through a production-inventory case study, illustrating its robustness against inference attacks and its superiority compared to a truncated Gaussian mechanism.



## **23. An Optimizable Suffix Is Worth A Thousand Templates: Efficient Black-box Jailbreaking without Affirmative Phrases via LLM as Optimizer**

cs.AI

Be accepeted as NAACL2025 Findings

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2408.11313v2) [paper-pdf](http://arxiv.org/pdf/2408.11313v2)

**Authors**: Weipeng Jiang, Zhenting Wang, Juan Zhai, Shiqing Ma, Zhengyu Zhao, Chao Shen

**Abstract**: Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.



## **24. Leveraging Generalizability of Image-to-Image Translation for Enhanced Adversarial Defense**

cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01399v1) [paper-pdf](http://arxiv.org/pdf/2504.01399v1)

**Authors**: Haibo Zhang, Zhihua Yao, Kouichi Sakurai, Takeshi Saitoh

**Abstract**: In the rapidly evolving field of artificial intelligence, machine learning emerges as a key technology characterized by its vast potential and inherent risks. The stability and reliability of these models are important, as they are frequent targets of security threats. Adversarial attacks, first rigorously defined by Ian Goodfellow et al. in 2013, highlight a critical vulnerability: they can trick machine learning models into making incorrect predictions by applying nearly invisible perturbations to images. Although many studies have focused on constructing sophisticated defensive mechanisms to mitigate such attacks, they often overlook the substantial time and computational costs of training and maintaining these models. Ideally, a defense method should be able to generalize across various, even unseen, adversarial attacks with minimal overhead. Building on our previous work on image-to-image translation-based defenses, this study introduces an improved model that incorporates residual blocks to enhance generalizability. The proposed method requires training only a single model, effectively defends against diverse attack types, and is well-transferable between different target models. Experiments show that our model can restore the classification accuracy from near zero to an average of 72\% while maintaining competitive performance compared to state-of-the-art methods.



## **25. Adversarial Example Soups: Improving Transferability and Stealthiness for Free**

cs.CV

Accepted by TIFS 2025

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2402.18370v3) [paper-pdf](http://arxiv.org/pdf/2402.18370v3)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yulong Yang, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: Transferable adversarial examples cause practical security risks since they can mislead a target model without knowing its internal knowledge. A conventional recipe for maximizing transferability is to keep only the optimal adversarial example from all those obtained in the optimization pipeline. In this paper, for the first time, we revisit this convention and demonstrate that those discarded, sub-optimal adversarial examples can be reused to boost transferability. Specifically, we propose ``Adversarial Example Soups'' (AES), with AES-tune for averaging discarded adversarial examples in hyperparameter tuning and AES-rand for stability testing. In addition, our AES is inspired by ``model soups'', which averages weights of multiple fine-tuned models for improved accuracy without increasing inference time. Extensive experiments validate the global effectiveness of our AES, boosting 10 state-of-the-art transfer attacks and their combinations by up to 13\% against 10 diverse (defensive) target models. We also show the possibility of generalizing AES to other types, \textit{e.g.}, directly averaging multiple in-the-wild adversarial examples that yield comparable success. A promising byproduct of AES is the improved stealthiness of adversarial examples since the perturbation variances are naturally reduced.



## **26. Breaking BERT: Gradient Attack on Twitter Sentiment Analysis for Targeted Misclassification**

cs.CL

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01345v1) [paper-pdf](http://arxiv.org/pdf/2504.01345v1)

**Authors**: Akil Raj Subedi, Taniya Shah, Aswani Kumar Cherukuri, Thanos Vasilakos

**Abstract**: Social media platforms like Twitter have increasingly relied on Natural Language Processing NLP techniques to analyze and understand the sentiments expressed in the user generated content. One such state of the art NLP model is Bidirectional Encoder Representations from Transformers BERT which has been widely adapted in sentiment analysis. BERT is susceptible to adversarial attacks. This paper aims to scrutinize the inherent vulnerabilities of such models in Twitter sentiment analysis. It aims to formulate a framework for constructing targeted adversarial texts capable of deceiving these models, while maintaining stealth. In contrast to conventional methodologies, such as Importance Reweighting, this framework core idea resides in its reliance on gradients to prioritize the importance of individual words within the text. It uses a whitebox approach to attain fine grained sensitivity, pinpointing words that exert maximal influence on the classification outcome. This paper is organized into three interdependent phases. It starts with fine-tuning a pre-trained BERT model on Twitter data. It then analyzes gradients of the model to rank words on their importance, and iteratively replaces those with feasible candidates until an acceptable solution is found. Finally, it evaluates the effectiveness of the adversarial text against the custom trained sentiment classification model. This assessment would help in gauging the capacity of the adversarial text to successfully subvert classification without raising any alarm.



## **27. STEREO: A Two-Stage Framework for Adversarially Robust Concept Erasing from Text-to-Image Diffusion Models**

cs.CV

Accepted to CVPR-2025. Code:  https://github.com/koushiksrivats/robust-concept-erasing

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2408.16807v2) [paper-pdf](http://arxiv.org/pdf/2408.16807v2)

**Authors**: Koushik Srivatsan, Fahad Shamshad, Muzammal Naseer, Vishal M. Patel, Karthik Nandakumar

**Abstract**: The rapid proliferation of large-scale text-to-image diffusion (T2ID) models has raised serious concerns about their potential misuse in generating harmful content. Although numerous methods have been proposed for erasing undesired concepts from T2ID models, they often provide a false sense of security; concept-erased models (CEMs) can still be manipulated via adversarial attacks to regenerate the erased concept. While a few robust concept erasure methods based on adversarial training have emerged recently, they compromise on utility (generation quality for benign concepts) to achieve robustness and/or remain vulnerable to advanced embedding space attacks. These limitations stem from the failure of robust CEMs to thoroughly search for "blind spots" in the embedding space. To bridge this gap, we propose STEREO, a novel two-stage framework that employs adversarial training as a first step rather than the only step for robust concept erasure. In the first stage, STEREO employs adversarial training as a vulnerability identification mechanism to search thoroughly enough. In the second robustly erase once stage, STEREO introduces an anchor-concept-based compositional objective to robustly erase the target concept in a single fine-tuning stage, while minimizing the degradation of model utility. We benchmark STEREO against seven state-of-the-art concept erasure methods, demonstrating its superior robustness to both white-box and black-box attacks, while largely preserving utility.



## **28. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01308v1) [paper-pdf](http://arxiv.org/pdf/2504.01308v1)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yichen Fu, Yichun Feng, Kin-man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.



## **29. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

cs.CR

23 pages, 7 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2401.07261v5) [paper-pdf](http://arxiv.org/pdf/2401.07261v5)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: Decentralized Finance (DeFi) incidents stemming from the exploitation of smart contract vulnerabilities have culminated in financial damages exceeding 3 billion US dollars. Existing defense mechanisms typically focus on detecting and reacting to malicious transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively. Based on the fact that most attack logic rely on deploying one or more intermediate smart contracts as supporting components to the exploitation of victim contracts, detection methods have been proposed that focus on identifying these adversarial contracts instead of adversarial transactions. However, previous state-of-the-art approaches in this direction have failed to produce results satisfactory enough for real-world deployment. In this paper, we propose a new framework for effectively detecting DeFi attacks via unveiling adversarial contracts. Our approach allows us to leverage common attack patterns, code semantics and intrinsic characteristics found in malicious smart contracts to build the LookAhead system based on Machine Learning (ML) classifiers and a transformer model that is able to effectively distinguish adversarial contracts from benign ones, and make timely predictions of different types of potential attacks. Experiments show that LookAhead achieves an F1-score as high as 0.8966, which represents an improvement of over 44.4% compared to the previous state-of-the-art solution Forta, with a False Positive Rate (FPR) at only 0.16%.



## **30. Strategize Globally, Adapt Locally: A Multi-Turn Red Teaming Agent with Dual-Level Learning**

cs.AI

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01278v1) [paper-pdf](http://arxiv.org/pdf/2504.01278v1)

**Authors**: Si Chen, Xiao Yu, Ninareh Mehrabi, Rahul Gupta, Zhou Yu, Ruoxi Jia

**Abstract**: The exploitation of large language models (LLMs) for malicious purposes poses significant security risks as these models become more powerful and widespread. While most existing red-teaming frameworks focus on single-turn attacks, real-world adversaries typically operate in multi-turn scenarios, iteratively probing for vulnerabilities and adapting their prompts based on threat model responses. In this paper, we propose \AlgName, a novel multi-turn red-teaming agent that emulates sophisticated human attackers through complementary learning dimensions: global tactic-wise learning that accumulates knowledge over time and generalizes to new attack goals, and local prompt-wise learning that refines implementations for specific goals when initial attempts fail. Unlike previous multi-turn approaches that rely on fixed strategy sets, \AlgName enables the agent to identify new jailbreak tactics, develop a goal-based tactic selection framework, and refine prompt formulations for selected tactics. Empirical evaluations on JailbreakBench demonstrate our framework's superior performance, achieving over 90\% attack success rates against GPT-3.5-Turbo and Llama-3.1-70B within 5 conversation turns, outperforming state-of-the-art baselines. These results highlight the effectiveness of dynamic learning in identifying and exploiting model vulnerabilities in realistic multi-turn scenarios.



## **31. Towards Resilient Federated Learning in CyberEdge Networks: Recent Advances and Future Trends**

cs.CR

15 pages, 8 figures, 4 tables, 122 references, journal paper

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01240v1) [paper-pdf](http://arxiv.org/pdf/2504.01240v1)

**Authors**: Kai Li, Zhengyang Zhang, Azadeh Pourkabirian, Wei Ni, Falko Dressler, Ozgur B. Akan

**Abstract**: In this survey, we investigate the most recent techniques of resilient federated learning (ResFL) in CyberEdge networks, focusing on joint training with agglomerative deduction and feature-oriented security mechanisms. We explore adaptive hierarchical learning strategies to tackle non-IID data challenges, improving scalability and reducing communication overhead. Fault tolerance techniques and agglomerative deduction mechanisms are studied to detect unreliable devices, refine model updates, and enhance convergence stability. Unlike existing FL security research, we comprehensively analyze feature-oriented threats, such as poisoning, inference, and reconstruction attacks that exploit model features. Moreover, we examine resilient aggregation techniques, anomaly detection, and cryptographic defenses, including differential privacy and secure multi-party computation, to strengthen FL security. In addition, we discuss the integration of 6G, large language models (LLMs), and interoperable learning frameworks to enhance privacy-preserving and decentralized cross-domain training. These advancements offer ultra-low latency, artificial intelligence (AI)-driven network management, and improved resilience against adversarial attacks, fostering the deployment of secure ResFL in CyberEdge networks.



## **32. TenAd: A Tensor-based Low-rank Black Box Adversarial Attack for Video Classification**

cs.CV

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01228v1) [paper-pdf](http://arxiv.org/pdf/2504.01228v1)

**Authors**: Kimia haghjooei, Mansoor Rezghi

**Abstract**: Deep learning models have achieved remarkable success in computer vision but remain vulnerable to adversarial attacks, particularly in black-box settings where model details are unknown. Existing adversarial attack methods(even those works with key frames) often treat video data as simple vectors, ignoring their inherent multi-dimensional structure, and require a large number of queries, making them inefficient and detectable. In this paper, we propose \textbf{TenAd}, a novel tensor-based low-rank adversarial attack that leverages the multi-dimensional properties of video data by representing videos as fourth-order tensors. By exploiting low-rank attack, our method significantly reduces the search space and the number of queries needed to generate adversarial examples in black-box settings. Experimental results on standard video classification datasets demonstrate that \textbf{TenAd} effectively generates imperceptible adversarial perturbations while achieving higher attack success rates and query efficiency compared to state-of-the-art methods. Our approach outperforms existing black-box adversarial attacks in terms of success rate, query efficiency, and perturbation imperceptibility, highlighting the potential of tensor-based methods for adversarial attacks on video models.



## **33. A Survey on Adversarial Contention Resolution**

cs.DC

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2403.03876v3) [paper-pdf](http://arxiv.org/pdf/2403.03876v3)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We also highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.



## **34. No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data**

cs.CR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2502.19537v3) [paper-pdf](http://arxiv.org/pdf/2502.19537v3)

**Authors**: Joshua Kazdan, Lisa Yu, Rylan Schaeffer, Chris Cundy, Sanmi Koyejo, Krishnamurthy Dvijotham

**Abstract**: Leading language model (LM) providers like OpenAI and Google offer fine-tuning APIs that allow customers to adapt LMs for specific use cases. To prevent misuse, these LM providers implement filtering mechanisms to block harmful fine-tuning data. Consequently, adversaries seeking to produce unsafe LMs via these APIs must craft adversarial training data that are not identifiably harmful. We make three contributions in this context: 1. We show that many existing attacks that use harmless data to create unsafe LMs rely on eliminating model refusals in the first few tokens of their responses. 2. We show that such prior attacks can be blocked by a simple defense that pre-fills the first few tokens from an aligned model before letting the fine-tuned model fill in the rest. 3. We describe a new data-poisoning attack, ``No, Of course I Can Execute'' (NOICE), which exploits an LM's formulaic refusal mechanism to elicit harmful responses. By training an LM to refuse benign requests on the basis of safety before fulfilling those requests regardless, we are able to jailbreak several open-source models and a closed-source model (GPT-4o). We show an attack success rate (ASR) of 57% against GPT-4o; our attack earned a Bug Bounty from OpenAI. Against open-source models protected by simple defenses, we improve ASRs by an average of 3.25 times compared to the best performing previous attacks that use only harmless data. NOICE demonstrates the exploitability of repetitive refusal mechanisms and broadens understanding of the threats closed-source models face from harmless data.



## **35. Multilingual and Multi-Accent Jailbreaking of Audio LLMs**

cs.SD

21 pages, 6 figures, 15 tables

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01094v1) [paper-pdf](http://arxiv.org/pdf/2504.01094v1)

**Authors**: Jaechul Roh, Virat Shejwalkar, Amir Houmansadr

**Abstract**: Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.



## **36. A Survey on Unlearnable Data**

cs.LG

31 pages, 3 figures, Code in  https://github.com/LiJiahao-Alex/Awesome-UnLearnable-Data

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2503.23536v2) [paper-pdf](http://arxiv.org/pdf/2503.23536v2)

**Authors**: Jiahao Li, Yiqiang Chen, Yunbing Xing, Yang Gu, Xiangyuan Lan

**Abstract**: Unlearnable data (ULD) has emerged as an innovative defense technique to prevent machine learning models from learning meaningful patterns from specific data, thus protecting data privacy and security. By introducing perturbations to the training data, ULD degrades model performance, making it difficult for unauthorized models to extract useful representations. Despite the growing significance of ULD, existing surveys predominantly focus on related fields, such as adversarial attacks and machine unlearning, with little attention given to ULD as an independent area of study. This survey fills that gap by offering a comprehensive review of ULD, examining unlearnable data generation methods, public benchmarks, evaluation metrics, theoretical foundations and practical applications. We compare and contrast different ULD approaches, analyzing their strengths, limitations, and trade-offs related to unlearnability, imperceptibility, efficiency and robustness. Moreover, we discuss key challenges, such as balancing perturbation imperceptibility with model degradation and the computational complexity of ULD generation. Finally, we highlight promising future research directions to advance the effectiveness and applicability of ULD, underscoring its potential to become a crucial tool in the evolving landscape of data protection in machine learning.



## **37. S3C2 Summit 2024-08: Government Secure Supply Chain Summit**

cs.CR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00924v1) [paper-pdf](http://arxiv.org/pdf/2504.00924v1)

**Authors**: Courtney Miller, William Enck, Yasemin Acar, Michel Cukier, Alexandros Kapravelos, Christian Kastner, Dominik Wermke, Laurie Williams

**Abstract**: Supply chain security has become a very important vector to consider when defending against adversary attacks. Due to this, more and more developers are keen on improving their supply chains to make them more robust against future threats. On August 29, 2024 researchers from the Secure Software Supply Chain Center (S3C2) gathered 14 practitioners from 10 government agencies to discuss the state of supply chain security. The goal of the summit is to share insights between companies and developers alike to foster new collaborations and ideas moving forward. Through this meeting, participants were questions on best practices and thoughts how to improve things for the future. In this paper we summarize the responses and discussions of the summit.



## **38. Alleviating Performance Disparity in Adversarial Spatiotemporal Graph Learning Under Zero-Inflated Distribution**

cs.LG

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00721v1) [paper-pdf](http://arxiv.org/pdf/2504.00721v1)

**Authors**: Songran Bai, Yuheng Ji, Yue Liu, Xingwei Zhang, Xiaolong Zheng, Daniel Dajun Zeng

**Abstract**: Spatiotemporal Graph Learning (SGL) under Zero-Inflated Distribution (ZID) is crucial for urban risk management tasks, including crime prediction and traffic accident profiling. However, SGL models are vulnerable to adversarial attacks, compromising their practical utility. While adversarial training (AT) has been widely used to bolster model robustness, our study finds that traditional AT exacerbates performance disparities between majority and minority classes under ZID, potentially leading to irreparable losses due to underreporting critical risk events. In this paper, we first demonstrate the smaller top-k gradients and lower separability of minority class are key factors contributing to this disparity. To address these issues, we propose MinGRE, a framework for Minority Class Gradients and Representations Enhancement. MinGRE employs a multi-dimensional attention mechanism to reweight spatiotemporal gradients, minimizing the gradient distribution discrepancies across classes. Additionally, we introduce an uncertainty-guided contrastive loss to improve the inter-class separability and intra-class compactness of minority representations with higher uncertainty. Extensive experiments demonstrate that the MinGRE framework not only significantly reduces the performance disparity across classes but also achieves enhanced robustness compared to existing baselines. These findings underscore the potential of our method in fostering the development of more equitable and robust models.



## **39. Impact of Data Duplication on Deep Neural Network-Based Image Classifiers: Robust vs. Standard Models**

cs.LG

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00638v1) [paper-pdf](http://arxiv.org/pdf/2504.00638v1)

**Authors**: Alireza Aghabagherloo, Aydin Abadi, Sumanta Sarkar, Vishnu Asutosh Dasu, Bart Preneel

**Abstract**: The accuracy and robustness of machine learning models against adversarial attacks are significantly influenced by factors such as training data quality, model architecture, the training process, and the deployment environment. In recent years, duplicated data in training sets, especially in language models, has attracted considerable attention. It has been shown that deduplication enhances both training performance and model accuracy in language models. While the importance of data quality in training image classifier Deep Neural Networks (DNNs) is widely recognized, the impact of duplicated images in the training set on model generalization and performance has received little attention.   In this paper, we address this gap and provide a comprehensive study on the effect of duplicates in image classification. Our analysis indicates that the presence of duplicated images in the training set not only negatively affects the efficiency of model training but also may result in lower accuracy of the image classifier. This negative impact of duplication on accuracy is particularly evident when duplicated data is non-uniform across classes or when duplication, whether uniform or non-uniform, occurs in the training set of an adversarially trained model. Even when duplicated samples are selected in a uniform way, increasing the amount of duplication does not lead to a significant improvement in accuracy.



## **40. Robust Recommender System: A Survey and Future Directions**

cs.IR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2309.02057v2) [paper-pdf](http://arxiv.org/pdf/2309.02057v2)

**Authors**: Kaike Zhang, Qi Cao, Fei Sun, Yunfan Wu, Shuchang Tao, Huawei Shen, Xueqi Cheng

**Abstract**: With the rapid growth of information, recommender systems have become integral for providing personalized suggestions and overcoming information overload. However, their practical deployment often encounters ``dirty'' data, where noise or malicious information can lead to abnormal recommendations. Research on improving recommender systems' robustness against such dirty data has thus gained significant attention. This survey provides a comprehensive review of recent work on recommender systems' robustness. We first present a taxonomy to organize current techniques for withstanding malicious attacks and natural noise. We then explore state-of-the-art methods in each category, including fraudster detection, adversarial training, certifiable robust training for defending against malicious attacks, and regularization, purification, self-supervised learning for defending against malicious attacks. Additionally, we summarize evaluation metrics and commonly used datasets for assessing robustness. We discuss robustness across varying recommendation scenarios and its interplay with other properties like accuracy, interpretability, privacy, and fairness. Finally, we delve into open issues and future research directions in this emerging field. Our goal is to provide readers with a comprehensive understanding of robust recommender systems and to identify key pathways for future research and development. To facilitate ongoing exploration, we maintain a continuously updated GitHub repository with related research: https://github.com/Kaike-Zhang/Robust-Recommender-System.



## **41. The Illusionist's Prompt: Exposing the Factual Vulnerabilities of Large Language Models with Linguistic Nuances**

cs.CL

work in progress

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.02865v1) [paper-pdf](http://arxiv.org/pdf/2504.02865v1)

**Authors**: Yining Wang, Yuquan Wang, Xi Li, Mi Zhang, Geng Hong, Min Yang

**Abstract**: As Large Language Models (LLMs) continue to advance, they are increasingly relied upon as real-time sources of information by non-expert users. To ensure the factuality of the information they provide, much research has focused on mitigating hallucinations in LLM responses, but only in the context of formal user queries, rather than maliciously crafted ones. In this study, we introduce The Illusionist's Prompt, a novel hallucination attack that incorporates linguistic nuances into adversarial queries, challenging the factual accuracy of LLMs against five types of fact-enhancing strategies. Our attack automatically generates highly transferrable illusory prompts to induce internal factual errors, all while preserving user intent and semantics. Extensive experiments confirm the effectiveness of our attack in compromising black-box LLMs, including commercial APIs like GPT-4o and Gemini-2.0, even with various defensive mechanisms.



## **42. Unleashing the Power of Pre-trained Encoders for Universal Adversarial Attack Detection**

cs.CV

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00429v1) [paper-pdf](http://arxiv.org/pdf/2504.00429v1)

**Authors**: Yinghe Zhang, Chi Liu, Shuai Zhou, Sheng Shen, Peng Gui

**Abstract**: Adversarial attacks pose a critical security threat to real-world AI systems by injecting human-imperceptible perturbations into benign samples to induce misclassification in deep learning models. While existing detection methods, such as Bayesian uncertainty estimation and activation pattern analysis, have achieved progress through feature engineering, their reliance on handcrafted feature design and prior knowledge of attack patterns limits generalization capabilities and incurs high engineering costs. To address these limitations, this paper proposes a lightweight adversarial detection framework based on the large-scale pre-trained vision-language model CLIP. Departing from conventional adversarial feature characterization paradigms, we innovatively adopt an anomaly detection perspective. By jointly fine-tuning CLIP's dual visual-text encoders with trainable adapter networks and learnable prompts, we construct a compact representation space tailored for natural images. Notably, our detection architecture achieves substantial improvements in generalization capability across both known and unknown attack patterns compared to traditional methods, while significantly reducing training overhead. This study provides a novel technical pathway for establishing a parameter-efficient and attack-agnostic defense paradigm, markedly enhancing the robustness of vision systems against evolving adversarial threats.



## **43. CopyQNN: Quantum Neural Network Extraction Attack under Varying Quantum Noise**

quant-ph

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00366v1) [paper-pdf](http://arxiv.org/pdf/2504.00366v1)

**Authors**: Zhenxiao Fu, Leyi Zhao, Xuhong Zhang, Yilun Xu, Gang Huang, Fan Chen

**Abstract**: Quantum Neural Networks (QNNs) have shown significant value across domains, with well-trained QNNs representing critical intellectual property often deployed via cloud-based QNN-as-a-Service (QNNaaS) platforms. Recent work has examined QNN model extraction attacks using classical and emerging quantum strategies. These attacks involve adversaries querying QNNaaS platforms to obtain labeled data for training local substitute QNNs that replicate the functionality of cloud-based models. However, existing approaches have largely overlooked the impact of varying quantum noise inherent in noisy intermediate-scale quantum (NISQ) computers, limiting their effectiveness in real-world settings. To address this limitation, we propose the CopyQNN framework, which employs a three-step data cleaning method to eliminate noisy data based on its noise sensitivity. This is followed by the integration of contrastive and transfer learning within the quantum domain, enabling efficient training of substitute QNNs using a limited but cleaned set of queried data. Experimental results on NISQ computers demonstrate that a practical implementation of CopyQNN significantly outperforms state-of-the-art QNN extraction attacks, achieving an average performance improvement of 8.73% across all tasks while reducing the number of required queries by 90x, with only a modest increase in hardware overhead.



## **44. System Identification from Partial Observations under Adversarial Attacks**

math.OC

9 pages, 2 figures

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2504.00244v1) [paper-pdf](http://arxiv.org/pdf/2504.00244v1)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper is concerned with the partially observed linear system identification, where the goal is to obtain reasonably accurate estimation of the balanced truncation of the true system up to the order $k$ from output measurements. We consider the challenging case of system identification under adversarial attacks, where the probability of having an attack at each time is $\Theta(1/k)$ while the value of the attack is arbitrary. We first show that the $l_1$-norm estimator exactly identifies the true Markov parameter matrix for nilpotent systems under any type of attack. We then build on this result to extend it to general systems and show that the estimation error exponentially decays as $k$ grows. The estimated balanced truncation model accordingly shows an exponentially decaying error for the identification of the true system up to the similarity transformation. This work is the first to provide the input-output analysis of the system with partial observations under arbitrary attacks.



## **45. Towards Adversarially Robust Dataset Distillation by Curvature Regularization**

cs.LG

14 pages, 3 figures

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2403.10045v3) [paper-pdf](http://arxiv.org/pdf/2403.10045v3)

**Authors**: Eric Xue, Yijiang Li, Haoyang Liu, Peiran Wang, Yifan Shen, Haohan Wang

**Abstract**: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable of generating robust distilled datasets that can withstand various adversarial attacks.



## **46. $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks**

cs.MA

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2504.00218v1) [paper-pdf](http://arxiv.org/pdf/2504.00218v1)

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Flemming, Tianlong Chen

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.



## **47. Backdoor Detection through Replicated Execution of Outsourced Training**

cs.CR

Published in the 3rd IEEE Conference on Secure and Trustworthy  Machine Learning (IEEE SaTML 2025)

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2504.00170v1) [paper-pdf](http://arxiv.org/pdf/2504.00170v1)

**Authors**: Hengrui Jia, Sierra Wyllie, Akram Bin Sediq, Ahmed Ibrahim, Nicolas Papernot

**Abstract**: It is common practice to outsource the training of machine learning models to cloud providers. Clients who do so gain from the cloud's economies of scale, but implicitly assume trust: the server should not deviate from the client's training procedure. A malicious server may, for instance, seek to insert backdoors in the model. Detecting a backdoored model without prior knowledge of both the backdoor attack and its accompanying trigger remains a challenging problem. In this paper, we show that a client with access to multiple cloud providers can replicate a subset of training steps across multiple servers to detect deviation from the training procedure in a similar manner to differential testing. Assuming some cloud-provided servers are benign, we identify malicious servers by the substantial difference between model updates required for backdooring and those resulting from clean training. Perhaps the strongest advantage of our approach is its suitability to clients that have limited-to-no local compute capability to perform training; we leverage the existence of multiple cloud providers to identify malicious updates without expensive human labeling or heavy computation. We demonstrate the capabilities of our approach on an outsourced supervised learning task where $50\%$ of the cloud providers insert their own backdoor; our approach is able to correctly identify $99.6\%$ of them. In essence, our approach is successful because it replaces the signature-based paradigm taken by existing approaches with an anomaly-based detection paradigm. Furthermore, our approach is robust to several attacks from adaptive adversaries utilizing knowledge of our detection scheme.



## **48. Privacy-Preserving Secure Neighbor Discovery for Wireless Networks**

cs.CR

10 pages, 6 figures. Author's version; accepted and presented at the  IEEE 23rd International Conference on Trust, Security and Privacy in  Computing and Communications (TrustCom) 2024

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2503.22232v2) [paper-pdf](http://arxiv.org/pdf/2503.22232v2)

**Authors**: Ahmed Mohamed Hussain, Panos Papadimitratos

**Abstract**: Traditional Neighbor Discovery (ND) and Secure Neighbor Discovery (SND) are key elements for network functionality. SND is a hard problem, satisfying not only typical security properties (authentication, integrity) but also verification of direct communication, which involves distance estimation based on time measurements and device coordinates. Defeating relay attacks, also known as "wormholes", leading to stealthy Byzantine links and significant degradation of communication and adversarial control, is key in many wireless networked systems. However, SND is not concerned with privacy; it necessitates revealing the identity and location of the device(s) participating in the protocol execution. This can be a deterrent for deployment, especially involving user-held devices in the emerging Internet of Things (IoT) enabled smart environments. To address this challenge, we present a novel Privacy-Preserving Secure Neighbor Discovery (PP-SND) protocol, enabling devices to perform SND without revealing their actual identities and locations, effectively decoupling discovery from the exposure of sensitive information. We use Homomorphic Encryption (HE) for computing device distances without revealing their actual coordinates, as well as employing a pseudonymous device authentication to hide identities while preserving communication integrity. PP-SND provides SND [1] along with pseudonymity, confidentiality, and unlinkability. Our presentation here is not specific to one wireless technology, and we assess the performance of the protocols (cryptographic overhead) on a Raspberry Pi 4 and provide a security and privacy analysis.



## **49. Adversarially Robust Learning with Optimal Transport Regularized Divergences**

cs.LG

33 pages, 2 figures

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2309.03791v2) [paper-pdf](http://arxiv.org/pdf/2309.03791v2)

**Authors**: Jeremiah Birrell, Reza Ebrahimi

**Abstract**: We introduce a new class of optimal-transport-regularized divergences, $D^c$, constructed via an infimal convolution between an information divergence, $D$, and an optimal-transport (OT) cost, $C$, and study their use in distributionally robust optimization (DRO). In particular, we propose the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These DRO-based methods are defined by minimizing the maximum expected loss over a $D^c$-neighborhood of the empirical distribution of the training data. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence; the addition of a principled and dynamical adversarial re-weighting on top of adversarial sample transport is a key innovation of $ARMOR_D$. $ARMOR_D$ can be viewed as a generalization of the best-performing loss functions and OT costs in the adversarial training literature; we demonstrate this flexibility by using $ARMOR_D$ to augment the UDR, TRADES, and MART methods and obtain improved performance on CIFAR-10 and CIFAR-100 image recognition. Specifically, augmenting with $ARMOR_D$ leads to 1.9\% and 2.1\% improvement against AutoAttack, a powerful ensemble of adversarial attacks, on CIFAR-10 and CIFAR-100 respectively. To foster reproducibility, we made the code accessible at https://github.com/star-ailab/ARMOR.



## **50. Pay More Attention to the Robustness of Prompt for Instruction Data Mining**

cs.AI

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2503.24028v1) [paper-pdf](http://arxiv.org/pdf/2503.24028v1)

**Authors**: Qiang Wang, Dawei Feng, Xu Zhang, Ao Shen, Yang Xu, Bo Ding, Huaimin Wang

**Abstract**: Instruction tuning has emerged as a paramount method for tailoring the behaviors of LLMs. Recent work has unveiled the potential for LLMs to achieve high performance through fine-tuning with a limited quantity of high-quality instruction data. Building upon this approach, we further explore the impact of prompt's robustness on the selection of high-quality instruction data. This paper proposes a pioneering framework of high-quality online instruction data mining for instruction tuning, focusing on the impact of prompt's robustness on the data mining process. Our notable innovation, is to generate the adversarial instruction data by conducting the attack for the prompt of online instruction data. Then, we introduce an Adversarial Instruction-Following Difficulty metric to measure how much help the adversarial instruction data can provide to the generation of the corresponding response. Apart from it, we propose a novel Adversarial Instruction Output Embedding Consistency approach to select high-quality online instruction data. We conduct extensive experiments on two benchmark datasets to assess the performance. The experimental results serve to underscore the effectiveness of our proposed two methods. Moreover, the results underscore the critical practical significance of considering prompt's robustness.



