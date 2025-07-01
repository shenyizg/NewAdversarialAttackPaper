# Latest Adversarial Attack Papers
**update at 2025-07-01 11:02:59**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. SQUASH: A SWAP-Based Quantum Attack to Sabotage Hybrid Quantum Neural Networks**

quant-ph

Keywords: Quantum Machine Learning, Hybrid Quantum Neural Networks,  SWAP Test, Fidelity, Circuit-level Attack

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24081v1) [paper-pdf](http://arxiv.org/pdf/2506.24081v1)

**Authors**: Rahul Kumar, Wenqi Wei, Ying Mao, Junaid Farooq, Ying Wang, Juntao Chen

**Abstract**: We propose a circuit-level attack, SQUASH, a SWAP-Based Quantum Attack to sabotage Hybrid Quantum Neural Networks (HQNNs) for classification tasks. SQUASH is executed by inserting SWAP gate(s) into the variational quantum circuit of the victim HQNN. Unlike conventional noise-based or adversarial input attacks, SQUASH directly manipulates the circuit structure, leading to qubit misalignment and disrupting quantum state evolution. This attack is highly stealthy, as it does not require access to training data or introduce detectable perturbations in input states. Our results demonstrate that SQUASH significantly degrades classification performance, with untargeted SWAP attacks reducing accuracy by up to 74.08\% and targeted SWAP attacks reducing target class accuracy by up to 79.78\%. These findings reveal a critical vulnerability in HQNN implementations, underscoring the need for more resilient architectures against circuit-level adversarial interventions.



## **2. STACK: Adversarial Attacks on LLM Safeguard Pipelines**

cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24068v1) [paper-pdf](http://arxiv.org/pdf/2506.24068v1)

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.



## **3. Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies**

math.OC

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24048v1) [paper-pdf](http://arxiv.org/pdf/2506.24048v1)

**Authors**: Tim Roith, Leon Bungert, Philipp Wacker

**Abstract**: Consensus-based optimization (CBO) has established itself as an efficient gradient-free optimization scheme, with attractive mathematical properties, such as mean-field convergence results for non-convex loss functions. In this work, we study CBO in the context of closed-box adversarial attacks, which are imperceptible input perturbations that aim to fool a classifier, without accessing its gradient. Our contribution is to establish a connection between the so-called consensus hopping as introduced by Riedl et al. and natural evolution strategies (NES) commonly applied in the context of adversarial attacks and to rigorously relate both methods to gradient-based optimization schemes. Beyond that, we provide a comprehensive experimental study that shows that despite the conceptual similarities, CBO can outperform NES and other evolutionary strategies in certain scenarios.



## **4. Quickest Detection of Adversarial Attacks Against Correlated Equilibria**

cs.GT

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24040v1) [paper-pdf](http://arxiv.org/pdf/2506.24040v1)

**Authors**: Kiarash Kazari, Aris Kanellopoulos, György Dán

**Abstract**: We consider correlated equilibria in strategic games in an adversarial environment, where an adversary can compromise the public signal used by the players for choosing their strategies, while players aim at detecting a potential attack as soon as possible to avoid loss of utility. We model the interaction between the adversary and the players as a zero-sum game and we derive the maxmin strategies for both the defender and the attacker using the framework of quickest change detection. We define a class of adversarial strategies that achieve the optimal trade-off between attack impact and attack detectability and show that a generalized CUSUM scheme is asymptotically optimal for the detection of the attacks. Our numerical results on the Sioux-Falls benchmark traffic routing game show that the proposed detection scheme can effectively limit the utility loss by a potential adversary.



## **5. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.



## **6. Benchmarking Spiking Neural Network Learning Methods with Varying Locality**

cs.NE

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2402.01782v2) [paper-pdf](http://arxiv.org/pdf/2402.01782v2)

**Authors**: Jiaqi Lin, Sen Lu, Malyaban Bal, Abhronil Sengupta

**Abstract**: Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have been shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but come with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, given the implicitly recurrent nature of SNNs, this research investigates the influence of the addition of explicit recurrence to SNNs. We experimentally prove that the addition of explicit recurrent weights enhances the robustness of SNNs. We also investigate the performance of local learning methods under gradient and non-gradient-based adversarial attacks.



## **7. A Unified Framework for Stealthy Adversarial Generation via Latent Optimization and Transferability Enhancement**

cs.CV

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23676v1) [paper-pdf](http://arxiv.org/pdf/2506.23676v1)

**Authors**: Gaozheng Pei, Ke Ma, Dongpeng Zhang, Chengzhi Sun, Qianqian Xu, Qingming Huang

**Abstract**: Due to their powerful image generation capabilities, diffusion-based adversarial example generation methods through image editing are rapidly gaining popularity. However, due to reliance on the discriminative capability of the diffusion model, these diffusion-based methods often struggle to generalize beyond conventional image classification tasks, such as in Deepfake detection. Moreover, traditional strategies for enhancing adversarial example transferability are challenging to adapt to these methods. To address these challenges, we propose a unified framework that seamlessly incorporates traditional transferability enhancement strategies into diffusion model-based adversarial example generation via image editing, enabling their application across a wider range of downstream tasks. Our method won first place in the "1st Adversarial Attacks on Deepfake Detectors: A Challenge in the Era of AI-Generated Media" competition at ACM MM25, which validates the effectiveness of our approach.



## **8. Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack**

cs.CL

12 pages main text, 27 pages total including references and  appendices. 13 figures, 10 tables. Accepted for publication in the LNCS  proceedings of CLEF 2025 (Best-of-Labs track)

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23661v1) [paper-pdf](http://arxiv.org/pdf/2506.23661v1)

**Authors**: Arnisa Fazla, Lucas Krauter, David Guzman Piedrahita, Andrianos Michail

**Abstract**: We extend BeamAttack, an adversarial attack algorithm designed to evaluate the robustness of text classification systems through word-level modifications guided by beam search. Our extensions include support for word deletions and the option to skip substitutions, enabling the discovery of minimal modifications that alter model predictions. We also integrate LIME to better prioritize word replacements. Evaluated across multiple datasets and victim models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA framework, our approach achieves over a 99\% attack success rate while preserving the semantic and lexical similarity of the original texts. Through both quantitative and qualitative analysis, we highlight BeamAttack's effectiveness and its limitations. Our implementation is available at https://github.com/LucK1Y/BeamAttack



## **9. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23581v1) [paper-pdf](http://arxiv.org/pdf/2506.23581v1)

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.



## **10. Efficient Resource Allocation under Adversary Attacks: A Decomposition-Based Approach**

cs.DS

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23442v1) [paper-pdf](http://arxiv.org/pdf/2506.23442v1)

**Authors**: Mansoor Davoodi, Setareh Maghsudi

**Abstract**: We address the problem of allocating limited resources in a network under persistent yet statistically unknown adversarial attacks. Each node in the network may be degraded, but not fully disabled, depending on its available defensive resources. The objective is twofold: to minimize total system damage and to reduce cumulative resource allocation and transfer costs over time. We model this challenge as a bi-objective optimization problem and propose a decomposition-based solution that integrates chance-constrained programming with network flow optimization. The framework separates the problem into two interrelated subproblems: determining optimal node-level allocations across time slots, and computing efficient inter-node resource transfers. We theoretically prove the convergence of our method to the optimal solution that would be obtained with full statistical knowledge of the adversary. Extensive simulations demonstrate that our method efficiently learns the adversarial patterns and achieves substantial gains in minimizing both damage and operational costs, comparing three benchmark strategies under various parameter settings.



## **11. TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs**

cs.CL

ICML 2025

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23423v1) [paper-pdf](http://arxiv.org/pdf/2506.23423v1)

**Authors**: Felipe Nuti, Tim Franzmeyer, João Henriques

**Abstract**: Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa.



## **12. Enhancing Adversarial Robustness through Multi-Objective Representation Learning**

cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2410.01697v4) [paper-pdf](http://arxiv.org/pdf/2410.01697v4)

**Authors**: Sedjro Salomon Hotegni, Sebastian Peitz

**Abstract**: Deep neural networks (DNNs) are vulnerable to small adversarial perturbations, which are tiny changes to the input data that appear insignificant but cause the model to produce drastically different outputs. Many defense methods require modifying model architectures during evaluation or performing test-time data purification. This not only introduces additional complexity but is often architecture-dependent. We show, however, that robust feature learning during training can significantly enhance DNN robustness. We propose MOREL, a multi-objective approach that aligns natural and adversarial features using cosine similarity and multi-positive contrastive losses to encourage similar features for same-class inputs. Extensive experiments demonstrate that MOREL significantly improves robustness against both white-box and black-box attacks. Our code is available at https://github.com/salomonhotegni/MOREL



## **13. Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning**

cs.LG

15 pages, 8 main pages of text, 13 figures, 5 tables. Made for a  Neurips workshop on backdoor attacks - extended version

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2310.11594v3) [paper-pdf](http://arxiv.org/pdf/2310.11594v3)

**Authors**: Taejin Kim, Jiarui Li, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: The delicate equilibrium between user privacy and the ability to unleash the potential of distributed data is an important concern. Federated learning, which enables the training of collaborative models without sharing of data, has emerged as a privacy-centric solution. This approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data into the training process, as well as evasion attacks that aim to induce misclassifications at test time. Our research investigates the intersection of adversarial training, a common defense method against evasion attacks, and backdoor attacks within federated learning. We introduce Adversarial Robustness Unhardening (ARU), which is employed by a subset of adversarial clients to intentionally undermine model robustness during federated training, rendering models susceptible to a broader range of evasion attacks. We present extensive experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning and backdoor attacks. Our results show that ARU can substantially undermine adversarial training's ability to harden models against test-time evasion attacks, and that adversaries employing ARU can even evade robust aggregation defenses that often neutralize poisoning or backdoor attacks.



## **14. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2411.16782v3) [paper-pdf](http://arxiv.org/pdf/2411.16782v3)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.



## **15. MedLeak: Multimodal Medical Data Leakage in Secure Federated Learning with Crafted Models**

cs.LG

Accepted by the IEEE/ACM conference on Connected Health:  Applications, Systems and Engineering Technologies 2025 (CHASE'25)

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2407.09972v2) [paper-pdf](http://arxiv.org/pdf/2407.09972v2)

**Authors**: Shanghao Shi, Md Shahedul Haque, Abhijeet Parida, Chaoyu Zhang, Marius George Linguraru, Y. Thomas Hou, Syed Muhammad Anwar, Wenjing Lou

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine learning models while keeping their data local, making it ideal for collaborations among healthcare institutions on sensitive data. However, in this paper, we propose a novel privacy attack called MedLeak, which allows a malicious FL server to recover high-quality site-specific private medical data from the client model updates. MedLeak works by introducing an adversarially crafted model during the FL training process. Honest clients, unaware of the insidious changes in the published models, continue to send back their updates as per the standard FL protocol. Leveraging a novel analytical method, MedLeak can efficiently recover private client data from the aggregated parameter updates, eliminating costly optimization. In addition, the scheme relies solely on the aggregated updates, thus rendering secure aggregation protocols ineffective, as they depend on the randomization of intermediate results for security while leaving the final aggregated results unaltered.   We implement MedLeak on medical image datasets (MedMNIST, COVIDx CXR-4, and Kaggle Brain Tumor MRI), as well as a medical text dataset (MedAbstract). The results demonstrate that our attack achieves high recovery rates and strong quantitative scores on both image and text datasets. We also thoroughly evaluate MedLeak across different attack parameters, providing insights into key factors that influence attack performance and potential defenses. Furthermore, we demonstrate that the recovered data can support downstream tasks such as disease classification with minimal performance loss. Our findings validate the need for enhanced privacy measures in FL systems, particularly for safeguarding sensitive medical data against powerful model inversion attacks.



## **16. Securing AI Systems: A Guide to Known Attacks and Impacts**

cs.CR

34 pages, 16 figures

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23296v1) [paper-pdf](http://arxiv.org/pdf/2506.23296v1)

**Authors**: Naoto Kiribuchi, Kengo Zenitani, Takayuki Semitsu

**Abstract**: Embedded into information systems, artificial intelligence (AI) faces security threats that exploit AI-specific vulnerabilities. This paper provides an accessible overview of adversarial attacks unique to predictive and generative AI systems. We identify eleven major attack types and explicitly link attack techniques to their impacts -- including information leakage, system compromise, and resource exhaustion -- mapped to the confidentiality, integrity, and availability (CIA) security triad. We aim to equip researchers, developers, security practitioners, and policymakers, even those without specialized AI security expertise, with foundational knowledge to recognize AI-specific risks and implement effective defenses, thereby enhancing the overall security posture of AI systems.



## **17. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

cs.CR

29 pages, 15 figures, 6 tables

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23260v1) [paper-pdf](http://arxiv.org/pdf/2506.23260v1)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces have dramatically expanded capabilities for real-time data retrieval, complex computation, and multi-step orchestration. Yet, the explosive proliferation of plugins, connectors, and inter-agent protocols has outpaced discovery mechanisms and security practices, resulting in brittle integrations vulnerable to diverse threats. In this survey, we introduce the first unified, end-to-end threat model for LLM-agent ecosystems, spanning host-to-tool and agent-to-agent communications, formalize adversary capabilities and attacker objectives, and catalog over thirty attack techniques. Specifically, we organized the threat model into four domains: Input Manipulation (e.g., prompt injections, long-context hijacks, multimodal adversarial inputs), Model Compromise (e.g., prompt- and parameter-level backdoors, composite and encrypted multi-backdoors, poisoning strategies), System and Privacy Attacks (e.g., speculative side-channels, membership inference, retrieval poisoning, social-engineering simulations), and Protocol Vulnerabilities (e.g., exploits in Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent Network Protocol (ANP), and Agent-to-Agent (A2A) protocol). For each category, we review representative scenarios, assess real-world feasibility, and evaluate existing defenses. Building on our threat taxonomy, we identify key open challenges and future research directions, such as securing MCP deployments through dynamic trust management and cryptographic provenance tracking; designing and hardening Agentic Web Interfaces; and achieving resilience in multi-agent and federated environments. Our work provides a comprehensive reference to guide the design of robust defense mechanisms and establish best practices for resilient LLM-agent workflows.



## **18. Fragile, Robust, and Antifragile: A Perspective from Parameter Responses in Reinforcement Learning Under Stress**

cs.LG

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.23036v1) [paper-pdf](http://arxiv.org/pdf/2506.23036v1)

**Authors**: Zain ul Abdeen, Ming Jin

**Abstract**: This paper explores Reinforcement learning (RL) policy robustness by systematically analyzing network parameters under internal and external stresses. Inspired by synaptic plasticity in neuroscience, synaptic filtering introduces internal stress by selectively perturbing parameters, while adversarial attacks apply external stress through modified agent observations. This dual approach enables the classification of parameters as fragile, robust, or antifragile, based on their influence on policy performance in clean and adversarial settings. Parameter scores are defined to quantify these characteristics, and the framework is validated on PPO-trained agents in Mujoco continuous control environments. The results highlight the presence of antifragile parameters that enhance policy performance under stress, demonstrating the potential of targeted filtering techniques to improve RL policy adaptability. These insights provide a foundation for future advancements in the design of robust and antifragile RL systems.



## **19. Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models**

cs.CV

Accepted to MLRC 2025

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22982v1) [paper-pdf](http://arxiv.org/pdf/2506.22982v1)

**Authors**: Atharv Mittal, Agam Pandey, Amritanshu Tiwari, Sukrit Jindal, Swadesh Swain

**Abstract**: Large Vision-Language Models (VLMs) have revolutionized computer vision, enabling tasks such as image classification, captioning, and visual question answering. However, they remain highly vulnerable to adversarial attacks, particularly in scenarios where both visual and textual modalities can be manipulated. In this study, we conduct a comprehensive reproducibility study of "An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models" validating the Cross-Prompt Attack (CroPA) and confirming its superior cross-prompt transferability compared to existing baselines. Beyond replication we propose several key improvements: (1) A novel initialization strategy that significantly improves Attack Success Rate (ASR). (2) Investigate cross-image transferability by learning universal perturbations. (3) A novel loss function targeting vision encoder attention mechanisms to improve generalization. Our evaluation across prominent VLMs -- including Flamingo, BLIP-2, and InstructBLIP as well as extended experiments on LLaVA validates the original results and demonstrates that our improvements consistently boost adversarial effectiveness. Our work reinforces the importance of studying adversarial vulnerabilities in VLMs and provides a more robust framework for generating transferable adversarial examples, with significant implications for understanding the security of VLMs in real-world applications.



## **20. VFEFL: Privacy-Preserving Federated Learning against Malicious Clients via Verifiable Functional Encryption**

cs.CR

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.12846v2) [paper-pdf](http://arxiv.org/pdf/2506.12846v2)

**Authors**: Nina Cai, Jinguang Han, Weizhi Meng

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods.



## **21. Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate**

cs.CV

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22806v1) [paper-pdf](http://arxiv.org/pdf/2506.22806v1)

**Authors**: Byung Hyun Lee, Sungjin Lim, Seunggyu Lee, Dong Un Kang, Se Young Chun

**Abstract**: Remarkable progress in text-to-image diffusion models has brought a major concern about potentially generating images on inappropriate or trademarked concepts. Concept erasing has been investigated with the goals of deleting target concepts in diffusion models while preserving other concepts with minimal distortion. To achieve these goals, recent concept erasing methods usually fine-tune the cross-attention layers of diffusion models. In this work, we first show that merely updating the cross-attention layers in diffusion models, which is mathematically equivalent to adding \emph{linear} modules to weights, may not be able to preserve diverse remaining concepts. Then, we propose a novel framework, dubbed Concept Pinpoint Eraser (CPE), by adding \emph{nonlinear} Residual Attention Gates (ResAGs) that selectively erase (or cut) target concepts while safeguarding remaining concepts from broad distributions by employing an attention anchoring loss to prevent the forgetting. Moreover, we adversarially train CPE with ResAG and learnable text embeddings in an iterative manner to maximize erasing performance and enhance robustness against adversarial attacks. Extensive experiments on the erasure of celebrities, artistic styles, and explicit contents demonstrated that the proposed CPE outperforms prior arts by keeping diverse remaining concepts while deleting the target concepts with robustness against attack prompts. Code is available at https://github.com/Hyun1A/CPE



## **22. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

cs.CL

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2407.01461v3) [paper-pdf](http://arxiv.org/pdf/2407.01461v3)

**Authors**: Xiaohua Wang, Zisu Huang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Qi Qian, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .



## **23. Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation**

cs.SE

13 pages, 6 figures

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22776v1) [paper-pdf](http://arxiv.org/pdf/2506.22776v1)

**Authors**: Sen Fang, Weiyuan Ding, Antonio Mastropaolo, Bowen Xu

**Abstract**: Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely unexplored.In this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies.



## **24. Kill Two Birds with One Stone! Trajectory enabled Unified Online Detection of Adversarial Examples and Backdoor Attacks**

cs.CR

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22722v1) [paper-pdf](http://arxiv.org/pdf/2506.22722v1)

**Authors**: Anmin Fu, Fanyu Meng, Huaibing Peng, Hua Ma, Zhi Zhang, Yifeng Zheng, Willy Susilo, Yansong Gao

**Abstract**: The proposed UniGuard is the first unified online detection framework capable of simultaneously addressing adversarial examples and backdoor attacks. UniGuard builds upon two key insights: first, both AE and backdoor attacks have to compromise the inference phase, making it possible to tackle them simultaneously during run-time via online detection. Second, an adversarial input, whether a perturbed sample in AE attacks or a trigger-carrying sample in backdoor attacks, exhibits distinctive trajectory signatures from a benign sample as it propagates through the layers of a DL model in forward inference. The propagation trajectory of the adversarial sample must deviate from that of its benign counterpart; otherwise, the adversarial objective cannot be fulfilled. Detecting these trajectory signatures is inherently challenging due to their subtlety; UniGuard overcomes this by treating the propagation trajectory as a time-series signal, leveraging LSTM and spectrum transformation to amplify differences between adversarial and benign trajectories that are subtle in the time domain. UniGuard exceptional efficiency and effectiveness have been extensively validated across various modalities (image, text, and audio) and tasks (classification and regression), ranging from diverse model architectures against a wide range of AE attacks and backdoor attacks, including challenging partial backdoors and dynamic triggers. When compared to SOTA methods, including ContraNet (NDSS 22) specific for AE detection and TED (IEEE SP 24) specific for backdoor detection, UniGuard consistently demonstrates superior performance, even when matched against each method's strengths in addressing their respective threats-each SOTA fails to parts of attack strategies while UniGuard succeeds for all.



## **25. VERA: Variational Inference Framework for Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22666v1) [paper-pdf](http://arxiv.org/pdf/2506.22666v1)

**Authors**: Anamika Lochab, Lu Yan, Patrick Pynadath, Xiangyu Zhang, Ruqi Zhang

**Abstract**: The rise of API-only access to state-of-the-art LLMs highlights the need for effective black-box jailbreak methods to identify model vulnerabilities in real-world settings. Without a principled objective for gradient-based optimization, most existing approaches rely on genetic algorithms, which are limited by their initialization and dependence on manually curated prompt pools. Furthermore, these methods require individual optimization for each prompt, failing to provide a comprehensive characterization of model vulnerabilities. To address this gap, we introduce VERA: Variational infErence fRamework for jAilbreaking. VERA casts black-box jailbreak prompting as a variational inference problem, training a small attacker LLM to approximate the target LLM's posterior over adversarial prompts. Once trained, the attacker can generate diverse, fluent jailbreak prompts for a target query without re-optimization. Experimental results show that VERA achieves strong performance across a range of target LLMs, highlighting the value of probabilistic inference for adversarial prompt generation.



## **26. MetaCipher: A General and Extensible Reinforcement Learning Framework for Obfuscation-Based Jailbreak Attacks on Black-Box LLMs**

cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22557v1) [paper-pdf](http://arxiv.org/pdf/2506.22557v1)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: The growing capabilities of large language models (LLMs) have exposed them to increasingly sophisticated jailbreak attacks. Among these, obfuscation-based attacks -- which encrypt malicious content to evade detection -- remain highly effective. By leveraging the reasoning ability of advanced LLMs to interpret encrypted prompts, such attacks circumvent conventional defenses that rely on keyword detection or context filtering. These methods are very difficult to defend against, as existing safety mechanisms are not designed to interpret or decode ciphered content. In this work, we propose \textbf{MetaCipher}, a novel obfuscation-based jailbreak framework, along with a reinforcement learning-based dynamic cipher selection mechanism that adaptively chooses optimal encryption strategies from a cipher pool. This approach enhances jailbreak effectiveness and generalizability across diverse task types, victim LLMs, and safety guardrails. Our framework is modular and extensible by design, supporting arbitrary cipher families and accommodating evolving adversarial strategies. We complement our method with a large-scale empirical analysis of cipher performance across multiple victim LLMs. Within as few as 10 queries, MetaCipher achieves over 92\% attack success rate (ASR) on most recent standard malicious prompt benchmarks against state-of-the-art non-reasoning LLMs, and over 74\% ASR against reasoning-capable LLMs, outperforming all existing obfuscation-based jailbreak methods. These results highlight the long-term robustness and adaptability of our approach, making it more resilient than prior methods in the face of advancing safety measures.



## **27. ARMOR: Robust Reinforcement Learning-based Control for UAVs under Physical Attacks**

cs.LG

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22423v1) [paper-pdf](http://arxiv.org/pdf/2506.22423v1)

**Authors**: Pritam Dash, Ethan Chan, Nathan P. Lawrence, Karthik Pattabiraman

**Abstract**: Unmanned Aerial Vehicles (UAVs) depend on onboard sensors for perception, navigation, and control. However, these sensors are susceptible to physical attacks, such as GPS spoofing, that can corrupt state estimates and lead to unsafe behavior. While reinforcement learning (RL) offers adaptive control capabilities, existing safe RL methods are ineffective against such attacks. We present ARMOR (Adaptive Robust Manipulation-Optimized State Representations), an attack-resilient, model-free RL controller that enables robust UAV operation under adversarial sensor manipulation. Instead of relying on raw sensor observations, ARMOR learns a robust latent representation of the UAV's physical state via a two-stage training framework. In the first stage, a teacher encoder, trained with privileged attack information, generates attack-aware latent states for RL policy training. In the second stage, a student encoder is trained via supervised learning to approximate the teacher's latent states using only historical sensor data, enabling real-world deployment without privileged information. Our experiments show that ARMOR outperforms conventional methods, ensuring UAV safety. Additionally, ARMOR improves generalization to unseen attacks and reduces training cost by eliminating the need for iterative adversarial training.



## **28. Secure Video Quality Assessment Resisting Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2410.06866v2) [paper-pdf](http://arxiv.org/pdf/2410.06866v2)

**Authors**: Ao-Xiang Zhang, Yuan-Gen Wang, Yu Ran, Weixuan Tang, Qingxiao Guan, Chunsheng Yang

**Abstract**: The exponential surge in video traffic has intensified the imperative for Video Quality Assessment (VQA). Leveraging cutting-edge architectures, current VQA models have achieved human-comparable accuracy. However, recent studies have revealed the vulnerability of existing VQA models against adversarial attacks. To establish a reliable and practical assessment system, a secure VQA model capable of resisting such malicious attacks is urgently demanded. Unfortunately, no attempt has been made to explore this issue. This paper first attempts to investigate general adversarial defense principles, aiming at endowing existing VQA models with security. Specifically, we first introduce random spatial grid sampling on the video frame for intra-frame defense. Then, we design pixel-wise randomization through a guardian map, globally neutralizing adversarial perturbations. Meanwhile, we extract temporal information from the video sequence as compensation for inter-frame defense. Building upon these principles, we present a novel VQA framework from the security-oriented perspective, termed SecureVQA. Extensive experiments indicate that SecureVQA sets a new benchmark in security while achieving competitive VQA performance compared with state-of-the-art models. Ablation studies delve deeper into analyzing the principles of SecureVQA, demonstrating their generalization and contributions to the security of leading VQA models.



## **29. A Self-scaled Approximate $\ell_0$ Regularization Robust Model for Outlier Detection**

eess.SP

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22277v1) [paper-pdf](http://arxiv.org/pdf/2506.22277v1)

**Authors**: Pengyang Song, Jue Wang

**Abstract**: Robust regression models in the presence of outliers have significant practical relevance in areas such as signal processing, financial econometrics, and energy management. Many existing robust regression methods, either grounded in statistical theory or sparse signal recovery, typically rely on the explicit or implicit assumption of outlier sparsity to filter anomalies and recover the underlying signal or data. However, these methods often suffer from limited robustness or high computational complexity, rendering them inefficient for large-scale problems. In this work, we propose a novel robust regression model based on a Self-scaled Approximate l0 Regularization Model (SARM) scheme. By introducing a self-scaling mechanism into the regularization term, the proposed model mitigates the negative impact of uneven or excessively large outlier magnitudes on robustness. We also develop an alternating minimization algorithm grounded in Proximal Operators and Block Coordinate Descent. We rigorously prove the algorithm convergence. Empirical comparisons with several state-of-the-art robust regression methods demonstrate that SARM not only achieves superior robustness but also significantly improves computational efficiency. Motivated by both the theoretical error bound and empirical observations, we further design a Two-Stage SARM (TSSARM) framework, which better utilizes sample information when the singular values of the design matrix are widely spread, thereby enhancing robustness under certain conditions. Finally, we validate our approach on a real-world load forecasting task. The experimental results show that our method substantially enhances the robustness of load forecasting against adversarial data attacks, which is increasingly critical in the era of heightened data security concerns.



## **30. Enhancing Object Detection Robustness: Detecting and Restoring Confidence in the Presence of Adversarial Patch Attacks**

cs.CV

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2403.12988v2) [paper-pdf](http://arxiv.org/pdf/2403.12988v2)

**Authors**: Roie Kazoom, Raz Birman, Ofer Hadar

**Abstract**: The widespread adoption of computer vision systems has underscored their susceptibility to adversarial attacks, particularly adversarial patch attacks on object detectors. This study evaluates defense mechanisms for the YOLOv5 model against such attacks. Optimized adversarial patches were generated and placed in sensitive image regions, by applying EigenCAM and grid search to determine optimal placement. We tested several defenses, including Segment and Complete (SAC), Inpainting, and Latent Diffusion Models. Our pipeline comprises three main stages: patch application, object detection, and defense analysis. Results indicate that adversarial patches reduce average detection confidence by 22.06\%. Defenses restored confidence levels by 3.45\% (SAC), 5.05\% (Inpainting), and significantly improved them by 26.61\%, which even exceeds the original accuracy levels, when using the Latent Diffusion Model, highlighting its superior effectiveness in mitigating the effects of adversarial patches.



## **31. Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses**

cs.CL

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21972v1) [paper-pdf](http://arxiv.org/pdf/2506.21972v1)

**Authors**: Mohamed Ahmed, Mohamed Abdelmouty, Mingyu Kim, Gunvanth Kandula, Alex Park, James C. Davis

**Abstract**: The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries.



## **32. Releasing Inequality Phenomenon in $\ell_{\infty}$-norm Adversarial Training via Input Gradient Distillation**

cs.CV

16 pages. Accepted by IEEE TIFS

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2305.09305v3) [paper-pdf](http://arxiv.org/pdf/2305.09305v3)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie, Jianhuang Lai

**Abstract**: Adversarial training (AT) is considered the most effective defense against adversarial attacks. However, a recent study revealed that \(\ell_{\infty}\)-norm adversarial training (\(\ell_{\infty}\)-AT) will also induce unevenly distributed input gradients, which is called the inequality phenomenon. This phenomenon makes the \(\ell_{\infty}\)-norm adversarially trained model more vulnerable than the standard-trained model when high-attribution or randomly selected pixels are perturbed, enabling robust and practical black-box attacks against \(\ell_{\infty}\)-adversarially trained models. In this paper, we propose a simple yet effective method called Input Gradient Distillation (IGD) to release the inequality phenomenon in $\ell_{\infty}$-AT. IGD distills the standard-trained teacher model's equal decision pattern into the $\ell_{\infty}$-adversarially trained student model by aligning input gradients of the student model and the standard-trained model with the Cosine Similarity. Experiments show that IGD can mitigate the inequality phenomenon and its threats while preserving adversarial robustness. Compared to vanilla $\ell_{\infty}$-AT, IGD reduces error rates against inductive noise, inductive occlusion, random noise, and noisy images in ImageNet-C by up to 60\%, 16\%, 50\%, and 21\%, respectively. Other than empirical experiments, we also conduct a theoretical analysis to explain why releasing the inequality phenomenon can improve such robustness and discuss why the severity of the inequality phenomenon varies according to the dataset's image resolution. Our code is available at https://github.com/fhdnskfbeuv/Inuput-Gradient-Distillation



## **33. One Video to Steal Them All: 3D-Printing IP Theft through Optical Side-Channels**

cs.CR

17 pages [Extended Version]

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21897v1) [paper-pdf](http://arxiv.org/pdf/2506.21897v1)

**Authors**: Twisha Chattopadhyay, Fabricio Ceschin, Marco E. Garza, Dymytriy Zyunkin, Animesh Chhotaray, Aaron P. Stebner, Saman Zonouz, Raheem Beyah

**Abstract**: The 3D printing industry is rapidly growing and increasingly adopted across various sectors including manufacturing, healthcare, and defense. However, the operational setup often involves hazardous environments, necessitating remote monitoring through cameras and other sensors, which opens the door to cyber-based attacks. In this paper, we show that an adversary with access to video recordings of the 3D printing process can reverse engineer the underlying 3D print instructions. Our model tracks the printer nozzle movements during the printing process and maps the corresponding trajectory into G-code instructions. Further, it identifies the correct parameters such as feed rate and extrusion rate, enabling successful intellectual property theft. To validate this, we design an equivalence checker that quantitatively compares two sets of 3D print instructions, evaluating their similarity in producing objects alike in shape, external appearance, and internal structure. Unlike simple distance-based metrics such as normalized mean square error, our equivalence checker is both rotationally and translationally invariant, accounting for shifts in the base position of the reverse engineered instructions caused by different camera positions. Our model achieves an average accuracy of 90.87 percent and generates 30.20 percent fewer instructions compared to existing methods, which often produce faulty or inaccurate prints. Finally, we demonstrate a fully functional counterfeit object generated by reverse engineering 3D print instructions from video.



## **34. On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling**

cs.CR

ACM Conference on Computer and Communications Security 2025

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21874v1) [paper-pdf](http://arxiv.org/pdf/2506.21874v1)

**Authors**: Stanley Wu, Ronik Bhaskar, Anna Yoo Jeong Ha, Shawn Shan, Haitao Zheng, Ben Y. Zhao

**Abstract**: Today's text-to-image generative models are trained on millions of images sourced from the Internet, each paired with a detailed caption produced by Vision-Language Models (VLMs). This part of the training pipeline is critical for supplying the models with large volumes of high-quality image-caption pairs during training. However, recent work suggests that VLMs are vulnerable to stealthy adversarial attacks, where adversarial perturbations are added to images to mislead the VLMs into producing incorrect captions.   In this paper, we explore the feasibility of adversarial mislabeling attacks on VLMs as a mechanism to poisoning training pipelines for text-to-image models. Our experiments demonstrate that VLMs are highly vulnerable to adversarial perturbations, allowing attackers to produce benign-looking images that are consistently miscaptioned by the VLM models. This has the effect of injecting strong "dirty-label" poison samples into the training pipeline for text-to-image models, successfully altering their behavior with a small number of poisoned samples. We find that while potential defenses can be effective, they can be targeted and circumvented by adaptive attackers. This suggests a cat-and-mouse game that is likely to reduce the quality of training data and increase the cost of text-to-image model development. Finally, we demonstrate the real-world effectiveness of these attacks, achieving high attack success (over 73%) even in black-box scenarios against commercial VLMs (Google Vertex AI and Microsoft Azure).



## **35. Adversarial Threats in Quantum Machine Learning: A Survey of Attacks and Defenses**

quant-ph

23 pages, 5 figures

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21842v1) [paper-pdf](http://arxiv.org/pdf/2506.21842v1)

**Authors**: Archisman Ghosh, Satwik Kundu, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) integrates quantum computing with classical machine learning, primarily to solve classification, regression and generative tasks. However, its rapid development raises critical security challenges in the Noisy Intermediate-Scale Quantum (NISQ) era. This chapter examines adversarial threats unique to QML systems, focusing on vulnerabilities in cloud-based deployments, hybrid architectures, and quantum generative models. Key attack vectors include model stealing via transpilation or output extraction, data poisoning through quantum-specific perturbations, reverse engineering of proprietary variational quantum circuits, and backdoor attacks. Adversaries exploit noise-prone quantum hardware and insufficiently secured QML-as-a-Service (QMLaaS) workflows to compromise model integrity, ownership, and functionality. Defense mechanisms leverage quantum properties to counter these threats. Noise signatures from training hardware act as non-invasive watermarks, while hardware-aware obfuscation techniques and ensemble strategies disrupt cloning attempts. Emerging solutions also adapt classical adversarial training and differential privacy to quantum settings, addressing vulnerabilities in quantum neural networks and generative architectures. However, securing QML requires addressing open challenges such as balancing noise levels for reliability and security, mitigating cross-platform attacks, and developing quantum-classical trust frameworks. This chapter summarizes recent advances in attacks and defenses, offering a roadmap for researchers and practitioners to build robust, trustworthy QML systems resilient to evolving adversarial landscapes.



## **36. Quantum Token Obfuscation via Superposition: A Post-Quantum Security Framework Using Multi-Basis Verification and Entropy-Driven Evolution**

quant-ph

16 pages, Significant revisions based on reviewer feedback

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2411.01252v3) [paper-pdf](http://arxiv.org/pdf/2411.01252v3)

**Authors**: S. M. Yousuf Iqbal Tomal, Abdullah Al Shafin

**Abstract**: Traditional cryptographic techniques, including token obfuscation, are increasingly vulnerable to quantum attacks due to advancements in quantum computing. Quantum algorithms such as Shor's and Grover's pose significant threats to classical security methods, necessitating quantum-resistant alternatives. This study proposes a quantum-based approach to token obfuscation that leverages superposition and multi-basis verification to enhance security against quantum adversaries. Tokens are encoded in quantum superposition states, ensuring probabilistic concealment until measured. A multi-basis verification protocol strengthens authentication by requiring validation across multiple quantum measurement bases. Additionally, a quantum decay protocol and token refresh mechanism dynamically manage the token lifecycle to prevent prolonged exposure and replay attacks. The model was tested through quantum simulations, evaluating entropy quality, adversarial robustness, and token verification reliability. Experimental validation demonstrates an entropy quality score of 0.9996, a 0% attack success rate across five adversarial models, and a 67% false positive rate, indicating strict security constraints. These findings confirm the effectiveness of quantum-based token obfuscation in preventing unauthorized reconstruction. The proposed approach provides a foundation for post-quantum cryptographic security by integrating entropy-driven state transformations, dynamic token evolution, and multi-basis verification. Future work will focus on optimizing computational efficiency and testing real-world implementations on quantum hardware.



## **37. PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks**

cs.CR

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2401.10586v2) [paper-pdf](http://arxiv.org/pdf/2401.10586v2)

**Authors**: Ping Guo, Xiang Li, Zhiyuan Yang, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defense mechanism, demonstrating significant improvements in robustness against query-based attacks.



## **38. A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns**

cs.CL

ACL 2025 Main

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2410.16155v2) [paper-pdf](http://arxiv.org/pdf/2410.16155v2)

**Authors**: Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: With the development of large language models, they are widely used as agents in various fields. A key component of agents is memory, which stores vital information but is susceptible to jailbreak attacks. Existing research mainly focuses on single-agent attacks and shared memory attacks. However, real-world scenarios often involve independent memory. In this paper, we propose the Troublemaker Makes Chaos in Honest Town (TMCHT) task, a large-scale, multi-agent, multi-topology text-based attack evaluation framework. TMCHT involves one attacker agent attempting to mislead an entire society of agents. We identify two major challenges in multi-agent attacks: (1) Non-complete graph structure, (2) Large-scale systems. We attribute these challenges to a phenomenon we term toxicity disappearing. To address these issues, we propose an Adversarial Replication Contagious Jailbreak (ARCJ) method, which optimizes the retrieval suffix to make poisoned samples more easily retrieved and optimizes the replication suffix to make poisoned samples have contagious ability. We demonstrate the superiority of our approach in TMCHT, with 23.51%, 18.95%, and 52.93% improvements in line topology, star topology, and 100-agent settings. Encourage community attention to the security of multi-agent systems.



## **39. Generative Adversarial Evasion and Out-of-Distribution Detection for UAV Cyber-Attacks**

cs.LG

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21142v1) [paper-pdf](http://arxiv.org/pdf/2506.21142v1)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: The growing integration of UAVs into civilian airspace underscores the need for resilient and intelligent intrusion detection systems (IDS), as traditional anomaly detection methods often fail to identify novel threats. A common approach treats unfamiliar attacks as out-of-distribution (OOD) samples; however, this leaves systems vulnerable when mitigation is inadequate. Moreover, conventional OOD detectors struggle to distinguish stealthy adversarial attacks from genuine OOD events. This paper introduces a conditional generative adversarial network (cGAN)-based framework for crafting stealthy adversarial attacks that evade IDS mechanisms. We first design a robust multi-class IDS classifier trained on benign UAV telemetry and known cyber-attacks, including Denial of Service (DoS), false data injection (FDI), man-in-the-middle (MiTM), and replay attacks. Using this classifier, our cGAN perturbs known attacks to generate adversarial samples that misclassify as benign while retaining statistical resemblance to OOD distributions. These adversarial samples are iteratively refined to achieve high stealth and success rates. To detect such perturbations, we implement a conditional variational autoencoder (CVAE), leveraging negative log-likelihood to separate adversarial inputs from authentic OOD samples. Comparative evaluation shows that CVAE-based regret scores significantly outperform traditional Mahalanobis distance-based detectors in identifying stealthy adversarial threats. Our findings emphasize the importance of advanced probabilistic modeling to strengthen IDS capabilities against adaptive, generative-model-based cyber intrusions.



## **40. Curriculum-Guided Antifragile Reinforcement Learning for Secure UAV Deconfliction under Observation-Space Attacks**

cs.LG

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21129v1) [paper-pdf](http://arxiv.org/pdf/2506.21129v1)

**Authors**: Deepak Kumar Panda, Adolfo Perrusquia, Weisi Guo

**Abstract**: Reinforcement learning (RL) policies deployed in safety-critical systems, such as unmanned aerial vehicle (UAV) navigation in dynamic airspace, are vulnerable to out-ofdistribution (OOD) adversarial attacks in the observation space. These attacks induce distributional shifts that significantly degrade value estimation, leading to unsafe or suboptimal decision making rendering the existing policy fragile. To address this vulnerability, we propose an antifragile RL framework designed to adapt against curriculum of incremental adversarial perturbations. The framework introduces a simulated attacker which incrementally increases the strength of observation-space perturbations which enables the RL agent to adapt and generalize across a wider range of OOD observations and anticipate previously unseen attacks. We begin with a theoretical characterization of fragility, formally defining catastrophic forgetting as a monotonic divergence in value function distributions with increasing perturbation strength. Building on this, we define antifragility as the boundedness of such value shifts and derive adaptation conditions under which forgetting is stabilized. Our method enforces these bounds through iterative expert-guided critic alignment using Wasserstein distance minimization across incrementally perturbed observations. We empirically evaluate the approach in a UAV deconfliction scenario involving dynamic 3D obstacles. Results show that the antifragile policy consistently outperforms standard and robust RL baselines when subjected to both projected gradient descent (PGD) and GPS spoofing attacks, achieving up to 15% higher cumulative reward and over 30% fewer conflict events. These findings demonstrate the practical and theoretical viability of antifragile reinforcement learning for secure and resilient decision-making in environments with evolving threat scenarios.



## **41. Robust Policy Switching for Antifragile Reinforcement Learning for UAV Deconfliction in Adversarial Environments**

cs.LG

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21127v1) [paper-pdf](http://arxiv.org/pdf/2506.21127v1)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: The increasing automation of navigation for unmanned aerial vehicles (UAVs) has exposed them to adversarial attacks that exploit vulnerabilities in reinforcement learning (RL) through sensor manipulation. Although existing robust RL methods aim to mitigate such threats, their effectiveness has limited generalization to out-of-distribution shifts from the optimal value distribution, as they are primarily designed to handle fixed perturbation. To address this limitation, this paper introduces an antifragile RL framework that enhances adaptability to broader distributional shifts by incorporating a switching mechanism based on discounted Thompson sampling (DTS). This mechanism dynamically selects among multiple robust policies to minimize adversarially induced state-action-value distribution shifts. The proposed approach first derives a diverse ensemble of action robust policies by accounting for a range of perturbations in the policy space. These policies are then modeled as a multiarmed bandit (MAB) problem, where DTS optimally selects policies in response to nonstationary Bernoulli rewards, effectively adapting to evolving adversarial strategies. Theoretical framework has also been provided where by optimizing the DTS to minimize the overall regrets due to distributional shift, results in effective adaptation against unseen adversarial attacks thus inducing antifragility. Extensive numerical simulations validate the effectiveness of the proposed framework in complex navigation environments with multiple dynamic three-dimensional obstacles and with stronger projected gradient descent (PGD) and spoofing attacks. Compared to conventional robust, non-adaptive RL methods, the antifragile approach achieves superior performance, demonstrating shorter navigation path lengths and a higher rate of conflict-free navigation trajectories compared to existing robust RL techniques



## **42. PhishKey: A Novel Centroid-Based Approach for Enhanced Phishing Detection Using Adaptive HTML Component Extraction**

cs.CR

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21106v1) [paper-pdf](http://arxiv.org/pdf/2506.21106v1)

**Authors**: Felipe Castaño, Eduardo Fidalgo, Enrique Alegre, Rocio Alaiz-Rodríguez, Raul Orduna, Francesco Zola

**Abstract**: Phishing attacks pose a significant cybersecurity threat, evolving rapidly to bypass detection mechanisms and exploit human vulnerabilities. This paper introduces PhishKey to address the challenges of adaptability, robustness, and efficiency. PhishKey is a novel phishing detection method using automatic feature extraction from hybrid sources. PhishKey combines character-level processing with Convolutional Neural Networks (CNN) for URL classification, and a Centroid-Based Key Component Phishing Extractor (CAPE) for HTML content at the word level. CAPE reduces noise and ensures complete sample processing avoiding crop operations on the input data. The predictions from both modules are integrated using a soft-voting ensemble to achieve more accurate and reliable classifications. Experimental evaluations on four state-of-the-art datasets demonstrate the effectiveness of PhishKey. It achieves up to 98.70% F1 Score and shows strong resistance to adversarial manipulations such as injection attacks with minimal performance degradation.



## **43. Boosting Generative Adversarial Transferability with Self-supervised Vision Transformer Features**

cs.CV

14 pages, 9 figures, to appear in ICCV 2025

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21046v1) [paper-pdf](http://arxiv.org/pdf/2506.21046v1)

**Authors**: Shangbo Wu, Yu-an Tan, Ruinan Ma, Wencong Ma, Dehua Zhu, Yuanzhang Li

**Abstract**: The ability of deep neural networks (DNNs) come from extracting and interpreting features from the data provided. By exploiting intermediate features in DNNs instead of relying on hard labels, we craft adversarial perturbation that generalize more effectively, boosting black-box transferability. These features ubiquitously come from supervised learning in previous work. Inspired by the exceptional synergy between self-supervised learning and the Transformer architecture, this paper explores whether exploiting self-supervised Vision Transformer (ViT) representations can improve adversarial transferability. We present dSVA -- a generative dual self-supervised ViT features attack, that exploits both global structural features from contrastive learning (CL) and local textural features from masked image modeling (MIM), the self-supervised learning paradigm duo for ViTs. We design a novel generative training framework that incorporates a generator to create black-box adversarial examples, and strategies to train the generator by exploiting joint features and the attention mechanism of self-supervised ViTs. Our findings show that CL and MIM enable ViTs to attend to distinct feature tendencies, which, when exploited in tandem, boast great adversarial generalizability. By disrupting dual deep features distilled by self-supervised ViTs, we are rewarded with remarkable black-box transferability to models of various architectures that outperform state-of-the-arts. Code available at https://github.com/spencerwooo/dSVA.



## **44. Doppelganger Method: Breaking Role Consistency in LLM Agent via Prompt-based Transferable Adversarial Attack**

cs.AI

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.14539v2) [paper-pdf](http://arxiv.org/pdf/2506.14539v2)

**Authors**: Daewon Kang, YeongHwan Shin, Doyeon Kim, Kyu-Hwan Jung, Meong Hi Son

**Abstract**: Since the advent of large language models, prompt engineering now enables the rapid, low-effort creation of diverse autonomous agents that are already in widespread use. Yet this convenience raises urgent concerns about the safety, robustness, and behavioral consistency of the underlying prompts, along with the pressing challenge of preventing those prompts from being exposed to user's attempts. In this paper, we propose the ''Doppelganger method'' to demonstrate the risk of an agent being hijacked, thereby exposing system instructions and internal information. Next, we define the ''Prompt Alignment Collapse under Adversarial Transfer (PACAT)'' level to evaluate the vulnerability to this adversarial transfer attack. We also propose a ''Caution for Adversarial Transfer (CAT)'' prompt to counter the Doppelganger method. The experimental results demonstrate that the Doppelganger method can compromise the agent's consistency and expose its internal information. In contrast, CAT prompts enable effective defense against this adversarial attack.



## **45. GuardSet-X: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset**

cs.CR

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.19054v2) [paper-pdf](http://arxiv.org/pdf/2506.19054v2)

**Authors**: Mintong Kang, Zhaorun Chen, Chejian Xu, Jiawei Zhang, Chengquan Guo, Minzhou Pan, Ivan Revilla, Yu Sun, Bo Li

**Abstract**: As LLMs become widespread across diverse applications, concerns about the security and safety of LLM interactions have intensified. Numerous guardrail models and benchmarks have been developed to ensure LLM content safety. However, existing guardrail benchmarks are often built upon ad hoc risk taxonomies that lack a principled grounding in standardized safety policies, limiting their alignment with real-world operational requirements. Moreover, they tend to overlook domain-specific risks, while the same risk category can carry different implications across different domains. To bridge these gaps, we introduce GuardSet-X, the first massive multi-domain safety policy-grounded guardrail dataset. GuardSet-X offers: (1) broad domain coverage across eight safety-critical domains, such as finance, law, and codeGen; (2) policy-grounded risk construction based on authentic, domain-specific safety guidelines; (3) diverse interaction formats, encompassing declarative statements, questions, instructions, and multi-turn conversations; (4) advanced benign data curation via detoxification prompting to challenge over-refusal behaviors; and (5) \textbf{attack-enhanced instances} that simulate adversarial inputs designed to bypass guardrails. Based on GuardSet-X, we benchmark 19 advanced guardrail models and uncover a series of findings, such as: (1) All models achieve varied F1 scores, with many demonstrating high variance across risk categories, highlighting their limited domain coverage and insufficient handling of domain-specific safety concerns; (2) As models evolve, their coverage of safety risks broadens, but performance on common risk categories may decrease; (3) All models remain vulnerable to optimized adversarial attacks. We believe that \dataset and the unique insights derived from our evaluations will advance the development of policy-aligned and resilient guardrail systems.



## **46. AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text**

cs.CL

This work has been submitted to NeurIPS 2025. Under review

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.22508v1) [paper-pdf](http://arxiv.org/pdf/2506.22508v1)

**Authors**: Chenyang Shao, Tianxing Li, Chenhao Pu, Fengli Xu, Yong Li

**Abstract**: In today's digital world, casual user-generated content often contains subtle cues that may inadvertently expose sensitive personal attributes. Such risks underscore the growing importance of effective text anonymization to safeguard individual privacy. However, existing methods either rely on rigid replacements that damage utility or cloud-based LLMs that are costly and pose privacy risks. To address these issues, we explore the use of locally deployed smaller-scale language models (SLMs) for anonymization. Yet training effective SLMs remains challenging due to limited high-quality supervision. To address the challenge, we propose AgentStealth, a self-reinforcing LLM anonymization framework.First, we introduce an adversarial anonymization workflow enhanced by In-context Contrastive Learning and Adaptive Utility-Aware Control. Second, we perform supervised adaptation of SLMs using high-quality data collected from the workflow, which includes both anonymization and attack signals. Finally, we apply online reinforcement learning where the model leverages its internal adversarial feedback to iteratively improve anonymization performance. Experiments on two datasets show that our method outperforms baselines in both anonymization effectiveness (+12.3%) and utility (+6.8%). Our lightweight design supports direct deployment on edge devices, avoiding cloud reliance and communication-based privacy risks. Our code is open-source at https://github.com/tsinghua-fib-lab/AgentStealth.



## **47. E-FreeM2: Efficient Training-Free Multi-Scale and Cross-Modal News Verification via MLLMs**

cs.MM

Accepted to AsiaCCS 2025 @ SCID

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.20944v1) [paper-pdf](http://arxiv.org/pdf/2506.20944v1)

**Authors**: Van-Hoang Phan, Long-Khanh Pham, Dang Vu, Anh-Duy Tran, Minh-Son Dao

**Abstract**: The rapid spread of misinformation in mobile and wireless networks presents critical security challenges. This study introduces a training-free, retrieval-based multimodal fact verification system that leverages pretrained vision-language models and large language models for credibility assessment. By dynamically retrieving and cross-referencing trusted data sources, our approach mitigates vulnerabilities of traditional training-based models, such as adversarial attacks and data poisoning. Additionally, its lightweight design enables seamless edge device integration without extensive on-device processing. Experiments on two fact-checking benchmarks achieve SOTA results, confirming its effectiveness in misinformation detection and its robustness against various attack vectors, highlighting its potential to enhance security in mobile and wireless communication environments.



## **48. SPA: Towards More Stealth and Persistent Backdoor Attacks in Federated Learning**

cs.CR

18 pages

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.20931v1) [paper-pdf](http://arxiv.org/pdf/2506.20931v1)

**Authors**: Chengcheng Zhu, Ye Li, Bosen Rao, Jiale Zhang, Yunlong Mao, Sheng Zhong

**Abstract**: Federated Learning (FL) has emerged as a leading paradigm for privacy-preserving distributed machine learning, yet the distributed nature of FL introduces unique security challenges, notably the threat of backdoor attacks. Existing backdoor strategies predominantly rely on end-to-end label supervision, which, despite their efficacy, often results in detectable feature disentanglement and limited persistence. In this work, we propose a novel and stealthy backdoor attack framework, named SPA, which fundamentally departs from traditional approaches by leveraging feature-space alignment rather than direct trigger-label association. Specifically, SPA reduces representational distances between backdoor trigger features and target class features, enabling the global model to misclassify trigger-embedded inputs with high stealth and persistence. We further introduce an adaptive, adversarial trigger optimization mechanism, utilizing boundary-search in the feature space to enhance attack longevity and effectiveness, even against defensive FL scenarios and non-IID data distributions. Extensive experiments on various FL benchmarks demonstrate that SPA consistently achieves high attack success rates with minimal impact on model utility, maintains robustness under challenging participation and data heterogeneity conditions, and exhibits persistent backdoor effects far exceeding those of conventional techniques. Our results call urgent attention to the evolving sophistication of backdoor threats in FL and emphasize the pressing need for advanced, feature-level defense techniques.



## **49. Empowering Digital Agriculture: A Privacy-Preserving Framework for Data Sharing and Collaborative Research**

cs.CR

arXiv admin note: text overlap with arXiv:2409.06069

**SubmitDate**: 2025-06-25    [abs](http://arxiv.org/abs/2506.20872v1) [paper-pdf](http://arxiv.org/pdf/2506.20872v1)

**Authors**: Osama Zafar, Rosemarie Santa González, Mina Namazi, Alfonso Morales, Erman Ayday

**Abstract**: Data-driven agriculture, which integrates technology and data into agricultural practices, has the potential to improve crop yield, disease resilience, and long-term soil health. However, privacy concerns, such as adverse pricing, discrimination, and resource manipulation, deter farmers from sharing data, as it can be used against them. To address this barrier, we propose a privacy-preserving framework that enables secure data sharing and collaboration for research and development while mitigating privacy risks. The framework combines dimensionality reduction techniques (like Principal Component Analysis (PCA)) and differential privacy by introducing Laplacian noise to protect sensitive information. The proposed framework allows researchers to identify potential collaborators for a target farmer and train personalized machine learning models either on the data of identified collaborators via federated learning or directly on the aggregated privacy-protected data. It also allows farmers to identify potential collaborators based on similarities. We have validated this on real-life datasets, demonstrating robust privacy protection against adversarial attacks and utility performance comparable to a centralized system. We demonstrate how this framework can facilitate collaboration among farmers and help researchers pursue broader research objectives. The adoption of the framework can empower researchers and policymakers to leverage agricultural data responsibly, paving the way for transformative advances in data-driven agriculture. By addressing critical privacy challenges, this work supports secure data integration, fostering innovation and sustainability in agricultural systems.



## **50. Universal and Efficient Detection of Adversarial Data through Nonuniform Impact on Network Layers**

cs.LG

arXiv admin note: substantial text overlap with arXiv:2410.17442

**SubmitDate**: 2025-06-25    [abs](http://arxiv.org/abs/2506.20816v1) [paper-pdf](http://arxiv.org/pdf/2506.20816v1)

**Authors**: Furkan Mumcu, Yasin Yilmaz

**Abstract**: Deep Neural Networks (DNNs) are notoriously vulnerable to adversarial input designs with limited noise budgets. While numerous successful attacks with subtle modifications to original input have been proposed, defense techniques against these attacks are relatively understudied. Existing defense approaches either focus on improving DNN robustness by negating the effects of perturbations or use a secondary model to detect adversarial data. Although equally important, the attack detection approach, which is studied in this work, provides a more practical defense compared to the robustness approach. We show that the existing detection methods are either ineffective against the state-of-the-art attack techniques or computationally inefficient for real-time processing. We propose a novel universal and efficient method to detect adversarial examples by analyzing the varying degrees of impact of attacks on different DNN layers. {Our method trains a lightweight regression model that predicts deeper-layer features from early-layer features, and uses the prediction error to detect adversarial samples.} Through theoretical arguments and extensive experiments, we demonstrate that our detection method is highly effective, computationally efficient for real-time processing, compatible with any DNN architecture, and applicable across different domains, such as image, video, and audio.



