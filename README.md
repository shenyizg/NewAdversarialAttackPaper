# Latest Adversarial Attack Papers
**update at 2025-06-18 11:07:13**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.12442v3) [paper-pdf](http://arxiv.org/pdf/2505.12442v3)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.



## **2. Variational Bayesian Bow tie Neural Networks with Shrinkage**

stat.ML

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2411.11132v3) [paper-pdf](http://arxiv.org/pdf/2411.11132v3)

**Authors**: Alisa Sheinkman, Sara Wade

**Abstract**: Despite the dominant role of deep models in machine learning, limitations persist, including overconfident predictions, susceptibility to adversarial attacks, and underestimation of variability in predictions. The Bayesian paradigm provides a natural framework to overcome such issues and has become the gold standard for uncertainty estimation with deep models, also providing improved accuracy and a framework for tuning critical hyperparameters. However, exact Bayesian inference is challenging, typically involving variational algorithms that impose strong independence and distributional assumptions. Moreover, existing methods are sensitive to the architectural choice of the network. We address these issues by focusing on a stochastic relaxation of the standard feed-forward rectified neural network and using sparsity-promoting priors on the weights of the neural network for increased robustness to architectural design. Thanks to Polya-Gamma data augmentation tricks, which render a conditionally linear and Gaussian model, we derive a fast, approximate variational inference algorithm that avoids distributional assumptions and independence across layers. Suitable strategies to further improve scalability and account for multimodality are considered.



## **3. Busting the Paper Ballot: Voting Meets Adversarial Machine Learning**

cs.CR

18 Pages. Author version of article to appear at CCS 2025

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14582v1) [paper-pdf](http://arxiv.org/pdf/2506.14582v1)

**Authors**: Kaleel Mahmood, Caleb Manicke, Ethan Rathbun, Aayushi Verma, Sohaib Ahmad, Nicholas Stamatakis, Laurent Michel, Benjamin Fuller

**Abstract**: We show the security risk associated with using machine learning classifiers in United States election tabulators. The central classification task in election tabulation is deciding whether a mark does or does not appear on a bubble associated to an alternative in a contest on the ballot. Barretto et al. (E-Vote-ID 2021) reported that convolutional neural networks are a viable option in this field, as they outperform simple feature-based classifiers.   Our contributions to election security can be divided into four parts. To demonstrate and analyze the hypothetical vulnerability of machine learning models on election tabulators, we first introduce four new ballot datasets. Second, we train and test a variety of different models on our new datasets. These models include support vector machines, convolutional neural networks (a basic CNN, VGG and ResNet), and vision transformers (Twins and CaiT). Third, using our new datasets and trained models, we demonstrate that traditional white box attacks are ineffective in the voting domain due to gradient masking. Our analyses further reveal that gradient masking is a product of numerical instability. We use a modified difference of logits ratio loss to overcome this issue (Croce and Hein, ICML 2020). Fourth, in the physical world, we conduct attacks with the adversarial examples generated using our new methods. In traditional adversarial machine learning, a high (50% or greater) attack success rate is ideal. However, for certain elections, even a 5% attack success rate can flip the outcome of a race. We show such an impact is possible in the physical domain. We thoroughly discuss attack realism, and the challenges and practicality associated with printing and scanning ballot adversarial examples.



## **4. Doppelgänger Method: Breaking Role Consistency in LLM Agent via Prompt-based Transferable Adversarial Attack**

cs.AI

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14539v1) [paper-pdf](http://arxiv.org/pdf/2506.14539v1)

**Authors**: Daewon Kang, YeongHwan Shin, Doyeon Kim, Kyu-Hwan Jung, Meong Hi Son

**Abstract**: Since the advent of large language models, prompt engineering now enables the rapid, low-effort creation of diverse autonomous agents that are already in widespread use. Yet this convenience raises urgent concerns about the safety, robustness, and behavioral consistency of the underlying prompts, along with the pressing challenge of preventing those prompts from being exposed to user's attempts. In this paper, we propose the ''Doppelg\"anger method'' to demonstrate the risk of an agent being hijacked, thereby exposing system instructions and internal information. Next, we define the ''Prompt Alignment Collapse under Adversarial Transfer (PACAT)'' level to evaluate the vulnerability to this adversarial transfer attack. We also propose a ''Caution for Adversarial Transfer (CAT)'' prompt to counter the Doppelg\"anger method. The experimental results demonstrate that the Doppelg\"anger method can compromise the agent's consistency and expose its internal information. In contrast, CAT prompts enable effective defense against this adversarial attack.



## **5. A Comparative Study on Proactive and Passive Detection of Deepfake Speech**

cs.SD

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14398v1) [paper-pdf](http://arxiv.org/pdf/2506.14398v1)

**Authors**: Chia-Hua Wu, Wanying Ge, Xin Wang, Junichi Yamagishi, Yu Tsao, Hsin-Min Wang

**Abstract**: Solutions for defending against deepfake speech fall into two categories: proactive watermarking models and passive conventional deepfake detectors. While both address common threats, their differences in training, optimization, and evaluation prevent a unified protocol for joint evaluation and selecting the best solutions for different cases. This work proposes a framework to evaluate both model types in deepfake speech detection. To ensure fair comparison and minimize discrepancies, all models were trained and tested on common datasets, with performance evaluated using a shared metric. We also analyze their robustness against various adversarial attacks, showing that different models exhibit distinct vulnerabilities to different speech attribute distortions. Our training and evaluation code is available at Github.



## **6. Excessive Reasoning Attack on Reasoning LLMs**

cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14374v1) [paper-pdf](http://arxiv.org/pdf/2506.14374v1)

**Authors**: Wai Man Si, Mingjie Li, Michael Backes, Yang Zhang

**Abstract**: Recent reasoning large language models (LLMs), such as OpenAI o1 and DeepSeek-R1, exhibit strong performance on complex tasks through test-time inference scaling. However, prior studies have shown that these models often incur significant computational costs due to excessive reasoning, such as frequent switching between reasoning trajectories (e.g., underthinking) or redundant reasoning on simple questions (e.g., overthinking). In this work, we expose a novel threat: adversarial inputs can be crafted to exploit excessive reasoning behaviors and substantially increase computational overhead without compromising model utility. Therefore, we propose a novel loss framework consisting of three components: (1) Priority Cross-Entropy Loss, a modification of the standard cross-entropy objective that emphasizes key tokens by leveraging the autoregressive nature of LMs; (2) Excessive Reasoning Loss, which encourages the model to initiate additional reasoning paths during inference; and (3) Delayed Termination Loss, which is designed to extend the reasoning process and defer the generation of final outputs. We optimize and evaluate our attack for the GSM8K and ORCA datasets on DeepSeek-R1-Distill-LLaMA and DeepSeek-R1-Distill-Qwen. Empirical results demonstrate a 3x to 9x increase in reasoning length with comparable utility performance. Furthermore, our crafted adversarial inputs exhibit transferability, inducing computational overhead in o3-mini, o1-mini, DeepSeek-R1, and QWQ models.



## **7. T2V-OptJail: Discrete Prompt Optimization for Text-to-Video Jailbreak Attacks**

cs.CV

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.06679v2) [paper-pdf](http://arxiv.org/pdf/2505.06679v2)

**Authors**: Jiayang Liu, Siyuan Liang, Shiqian Zhao, Rongcheng Tu, Wenbo Zhou, Aishan Liu, Dacheng Tao, Siew Kei Lam

**Abstract**: In recent years, fueled by the rapid advancement of diffusion models, text-to-video (T2V) generation models have achieved remarkable progress, with notable examples including Pika, Luma, Kling, and Open-Sora. Although these models exhibit impressive generative capabilities, they also expose significant security risks due to their vulnerability to jailbreak attacks, where the models are manipulated to produce unsafe content such as pornography, violence, or discrimination. Existing works such as T2VSafetyBench provide preliminary benchmarks for safety evaluation, but lack systematic methods for thoroughly exploring model vulnerabilities. To address this gap, we are the first to formalize the T2V jailbreak attack as a discrete optimization problem and propose a joint objective-based optimization framework, called T2V-OptJail. This framework consists of two key optimization goals: bypassing the built-in safety filtering mechanisms to increase the attack success rate, preserving semantic consistency between the adversarial prompt and the unsafe input prompt, as well as between the generated video and the unsafe input prompt, to enhance content controllability. In addition, we introduce an iterative optimization strategy guided by prompt variants, where multiple semantically equivalent candidates are generated in each round, and their scores are aggregated to robustly guide the search toward optimal adversarial prompts. We conduct large-scale experiments on several T2V models, covering both open-source models and real commercial closed-source models. The experimental results show that the proposed method improves 11.4% and 10.0% over the existing state-of-the-art method in terms of attack success rate assessed by GPT-4, attack success rate assessed by human accessors, respectively, verifying the significant advantages of the method in terms of attack effectiveness and content control.



## **8. RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?**

cs.LG

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14261v1) [paper-pdf](http://arxiv.org/pdf/2506.14261v1)

**Authors**: Rohan Gupta, Erik Jenner

**Abstract**: Latent-space monitors aim to detect undesirable behaviours in large language models by leveraging internal model representations rather than relying solely on black-box outputs. These methods have shown promise in identifying behaviours such as deception and unsafe completions, but a critical open question remains: can LLMs learn to evade such monitors? To study this, we introduce RL-Obfuscation, in which LLMs are finetuned via reinforcement learning to bypass latent-space monitors while maintaining coherent generations. We apply RL-Obfuscation to LLMs ranging from 7B to 14B parameters and evaluate evasion success against a suite of monitors. We find that token-level latent-space monitors are highly vulnerable to this attack. More holistic monitors, such as max-pooling or attention-based probes, remain robust. Moreover, we show that adversarial policies trained to evade a single static monitor generalise to unseen monitors of the same type. Finally, we study how the policy learned by RL bypasses these monitors and find that the model can also learn to repurpose tokens to mean something different internally.



## **9. CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement**

cs.CL

Accepted in ACL LLMSec Workshop 2025

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.12368v2) [paper-pdf](http://arxiv.org/pdf/2505.12368v2)

**Authors**: Gauri Kholkar, Ratinder Ahuja

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations. To demonstrate our framework's utility, we train CaptureGuard on our generated data. This new model drastically reduces both false negative and false positive rates on our context-aware datasets while also generalizing effectively to external benchmarks, establishing a path toward more robust and practical prompt injection defenses.



## **10. Mitigating Data Poisoning Attacks to Local Differential Privacy**

cs.CR

The 32nd ACM Conference on Computer and Communications Security (CCS  2025)

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.02156v2) [paper-pdf](http://arxiv.org/pdf/2506.02156v2)

**Authors**: Xiaolin Li, Ninghui Li, Boyang Wang, Wenhai Sun

**Abstract**: The distributed nature of local differential privacy (LDP) invites data poisoning attacks and poses unforeseen threats to the underlying LDP-supported applications. In this paper, we propose a comprehensive mitigation framework for popular frequency estimation, which contains a suite of novel defenses, including malicious user detection, attack pattern recognition, and damaged utility recovery. In addition to existing attacks, we explore new adaptive adversarial activities for our mitigation design. For detection, we present a new method to precisely identify bogus reports and thus LDP aggregation can be performed over the ``clean'' data. When the attack behavior becomes stealthy and direct filtering out malicious users is difficult, we further propose a detection that can effectively recognize hidden adversarial patterns, thus facilitating the decision-making of service providers. These detection methods require no additional data and attack information and incur minimal computational cost. Our experiment demonstrates their excellent performance and substantial improvement over previous work in various settings. In addition, we conduct an empirical analysis of LDP post-processing for corrupted data recovery and propose a new post-processing method, through which we reveal new insights into protocol recommendations in practice and key design principles for future research.



## **11. Weakest Link in the Chain: Security Vulnerabilities in Advanced Reasoning Models**

cs.AI

Accepted to LLMSEC 2025

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13726v1) [paper-pdf](http://arxiv.org/pdf/2506.13726v1)

**Authors**: Arjun Krishna, Aaditya Rastogi, Erick Galinkin

**Abstract**: The introduction of advanced reasoning capabilities have improved the problem-solving performance of large language models, particularly on math and coding benchmarks. However, it remains unclear whether these reasoning models are more or less vulnerable to adversarial prompt attacks than their non-reasoning counterparts. In this work, we present a systematic evaluation of weaknesses in advanced reasoning models compared to similar non-reasoning models across a diverse set of prompt-based attack categories. Using experimental data, we find that on average the reasoning-augmented models are \emph{slightly more robust} than non-reasoning models (42.51\% vs 45.53\% attack success rate, lower is better). However, this overall trend masks significant category-specific differences: for certain attack types the reasoning models are substantially \emph{more vulnerable} (e.g., up to 32 percentage points worse on a tree-of-attacks prompt), while for others they are markedly \emph{more robust} (e.g., 29.8 points better on cross-site scripting injection). Our findings highlight the nuanced security implications of advanced reasoning in language models and emphasize the importance of stress-testing safety across diverse adversarial techniques.



## **12. Deceptive Path Planning: A Bayesian Game Approach**

eess.SY

8 pages, 9 figures. This work has been submitted to the IEEE for  possible publication

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13650v1) [paper-pdf](http://arxiv.org/pdf/2506.13650v1)

**Authors**: Violetta Rostobaya, James Berneburg, Yue Guan, Michael Dorothy, Daigo Shishika

**Abstract**: This paper investigates how an autonomous agent can transmit information through its motion in an adversarial setting. We consider scenarios where an agent must reach its goal while deceiving an intelligent observer about its destination. We model this interaction as a dynamic Bayesian game between a mobile Attacker with a privately known goal and a Defender who infers the Attacker's intent to allocate defensive resources effectively. We use Perfect Bayesian Nash Equilibrium (PBNE) as our solution concept and propose a computationally efficient approach to find it. In the resulting equilibrium, the Defender employs a simple Markovian strategy, while the Attacker strategically balances deception and goal efficiency by stochastically mixing shortest and non-shortest paths to manipulate the Defender's beliefs. Numerical experiments demonstrate the advantages of our PBNE-based strategies over existing methods based on one-sided optimization.



## **13. Unlearning-Enhanced Website Fingerprinting Attack: Against Backdoor Poisoning in Anonymous Networks**

cs.CR

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13563v1) [paper-pdf](http://arxiv.org/pdf/2506.13563v1)

**Authors**: Yali Yuan, Kai Xu, Ruolin Ma, Yuchen Zhang

**Abstract**: Website Fingerprinting (WF) is an effective tool for regulating and governing the dark web. However, its performance can be significantly degraded by backdoor poisoning attacks in practical deployments. This paper aims to address the problem of hidden backdoor poisoning attacks faced by Website Fingerprinting attack, and designs a feasible mothed that integrates unlearning technology to realize detection of automatic poisoned points and complete removal of its destructive effects, requiring only a small number of known poisoned test points. Taking Tor onion routing as an example, our method evaluates the influence value of each training sample on these known poisoned test points as the basis for judgment. We optimize the use of influence scores to identify poisoned samples within the training dataset. Furthermore, by quantifying the difference between the contribution of model parameters on the taining data and the clean data, the target parameters are dynamically adjusted to eliminate the impact of the backdoor attacks. Experiments on public datasets under the assumptions of closed-world (CW) and open-world (OW) verify the effectiveness of the proposed method. In complex scenes containing both clean website fingerprinting features and backdoor triggers, the accuracy of the model on the poisoned dataset and the test dataset is stable at about 80%, significantly outperforming the traditional WF attack models. In addition, the proposed method achieves a 2-3 times speedup in runtime efficiency compared to baseline methods. By incorporating machine unlearning, we realize a WF attack model that exhibits enhanced resistance to backdoor poisoning and faster execution speeds in adversarial settings.



## **14. Fast Second-Order Online Kernel Learning through Incremental Matrix Sketching and Decomposition**

cs.LG

Accepted by IJCAI 2025

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2410.11188v2) [paper-pdf](http://arxiv.org/pdf/2410.11188v2)

**Authors**: Dongxie Wen, Xiao Zhang, Zhewei Wei, Chenping Hou, Shuai Li, Weinan Zhang

**Abstract**: Online Kernel Learning (OKL) has attracted considerable research interest due to its promising predictive performance in streaming environments. Second-order approaches are particularly appealing for OKL as they often offer substantial improvements in regret guarantees. However, existing second-order OKL approaches suffer from at least quadratic time complexity with respect to the pre-set budget, rendering them unsuitable for meeting the real-time demands of large-scale streaming recommender systems. The singular value decomposition required to obtain explicit feature mapping is also computationally expensive due to the complete decomposition process. Moreover, the absence of incremental updates to manage approximate kernel space causes these algorithms to perform poorly in adversarial environments and real-world streaming recommendation datasets. To address these issues, we propose FORKS, a fast incremental matrix sketching and decomposition approach tailored for second-order OKL. FORKS constructs an incremental maintenance paradigm for second-order kernelized gradient descent, which includes incremental matrix sketching for kernel approximation and incremental matrix decomposition for explicit feature mapping construction. Theoretical analysis demonstrates that FORKS achieves a logarithmic regret guarantee on par with other second-order approaches while maintaining a linear time complexity w.r.t. the budget, significantly enhancing efficiency over existing approaches. We validate the performance of FORKS through extensive experiments conducted on real-world streaming recommendation datasets, demonstrating its superior scalability and robustness against adversarial attacks.



## **15. Blockchain Address Poisoning**

cs.CR

To appear in Proceedings of the 34th USENIX Security Symposium  (USENIX Security'25)

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2501.16681v2) [paper-pdf](http://arxiv.org/pdf/2501.16681v2)

**Authors**: Taro Tsuchiya, Jin-Dong Dong, Kyle Soska, Nicolas Christin

**Abstract**: In many blockchains, e.g., Ethereum, Binance Smart Chain (BSC), the primary representation used for wallet addresses is a hardly memorable 40-digit hexadecimal string. As a result, users often select addresses from their recent transaction history, which enables blockchain address poisoning. The adversary first generates lookalike addresses similar to one with which the victim has previously interacted, and then engages with the victim to ``poison'' their transaction history. The goal is to have the victim mistakenly send tokens to the lookalike address, as opposed to the intended recipient. Compared to contemporary studies, this paper provides four notable contributions. First, we develop a detection system and perform measurements over two years on both Ethereum and BSC. We identify 13~times more attack attempts than reported previously -- totaling 270M on-chain attacks targeting 17M victims. 6,633 incidents have caused at least 83.8M USD in losses, which makes blockchain address poisoning one of the largest cryptocurrency phishing schemes observed in the wild. Second, we analyze a few large attack entities using improved clustering techniques, and model attacker profitability and competition. Third, we reveal attack strategies -- targeted populations, success conditions (address similarity, timing), and cross-chain attacks. Fourth, we mathematically define and simulate the lookalike address generation process across various software- and hardware-based implementations, and identify a large-scale attacker group that appears to use GPUs. We also discuss defensive countermeasures.



## **16. From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs**

cs.CR

10 pages

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13434v1) [paper-pdf](http://arxiv.org/pdf/2506.13434v1)

**Authors**: Alsharif Abuadbba, Chris Hicks, Kristen Moore, Vasilios Mavroudis, Burak Hasircioglu, Diksha Goel, Piers Jennings

**Abstract**: Large Language Models (LLMs) are set to reshape cybersecurity by augmenting red and blue team operations. Red teams can exploit LLMs to plan attacks, craft phishing content, simulate adversaries, and generate exploit code. Conversely, blue teams may deploy them for threat intelligence synthesis, root cause analysis, and streamlined documentation. This dual capability introduces both transformative potential and serious risks.   This position paper maps LLM applications across cybersecurity frameworks such as MITRE ATT&CK and the NIST Cybersecurity Framework (CSF), offering a structured view of their current utility and limitations. While LLMs demonstrate fluency and versatility across various tasks, they remain fragile in high-stakes, context-heavy environments. Key limitations include hallucinations, limited context retention, poor reasoning, and sensitivity to prompts, which undermine their reliability in operational settings.   Moreover, real-world integration raises concerns around dual-use risks, adversarial misuse, and diminished human oversight. Malicious actors could exploit LLMs to automate reconnaissance, obscure attack vectors, and lower the technical threshold for executing sophisticated attacks.   To ensure safer adoption, we recommend maintaining human-in-the-loop oversight, enhancing model explainability, integrating privacy-preserving mechanisms, and building systems robust to adversarial exploitation. As organizations increasingly adopt AI driven cybersecurity, a nuanced understanding of LLMs' risks and operational impacts is critical to securing their defensive value while mitigating unintended consequences.



## **17. Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks**

cs.AI

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13276v1) [paper-pdf](http://arxiv.org/pdf/2506.13276v1)

**Authors**: Yuefei Lyu, Chaozhuo Li, Xi Zhang, Tianle Zhang

**Abstract**: Text-attributed graphs (TAGs) integrate textual data with graph structures, providing valuable insights in applications such as social network analysis and recommendation systems. Graph Neural Networks (GNNs) effectively capture both topological structure and textual information in TAGs but are vulnerable to adversarial attacks. Existing graph injection attack (GIA) methods assume that attackers can directly manipulate the embedding layer, producing non-explainable node embeddings. Furthermore, the effectiveness of these attacks often relies on surrogate models with high training costs. Thus, this paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs. Our approach leverages large language models (LLMs) to generate interpretable text-level node attributes directly, ensuring attacks remain feasible in real-world scenarios. We design strategies for LLM prompting that balance exploration and reliability to guide text generation, and propose a similarity assessment method to evaluate attack text effectiveness in disrupting graph homophily. This method efficiently perturbs the target node with minimal training costs in a strict black-box setting, ensuring a text-level graph injection attack for TAGs. Experiments on real-world TAG datasets validate the superior performance of ATAG-LLM compared to state-of-the-art embedding-level and text-level attack methods.



## **18. CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models**

cs.CV

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2502.07225v2) [paper-pdf](http://arxiv.org/pdf/2502.07225v2)

**Authors**: Sen Peng, Mingyue Wang, Jianfei He, Jijia Yang, Xiaohua Jia

**Abstract**: Latent diffusion models have recently demonstrated superior capabilities in many downstream image synthesis tasks. However, customization of latent diffusion models using unauthorized data can severely compromise the privacy and intellectual property rights of data owners. Adversarial examples as protective perturbations have been developed to defend against unauthorized data usage by introducing imperceptible noise to customization samples, preventing diffusion models from effectively learning them. In this paper, we first reveal that the primary reason adversarial examples are effective as protective perturbations in latent diffusion models is the distortion of their latent representations, as demonstrated through qualitative and quantitative experiments. We then propose the Contrastive Adversarial Training (CAT) utilizing lightweight adapters as an adaptive attack against these protection methods, highlighting their lack of robustness. Extensive experiments demonstrate that our CAT method significantly reduces the effectiveness of protective perturbations in customization, urging the community to reconsider and improve the robustness of existing protective perturbations. The code is available at https://github.com/senp98/CAT.



## **19. Detecting Hard-Coded Credentials in Software Repositories via LLMs**

cs.CR

Accepted to the ACM Digital Threats: Research and Practice (DTRAP)

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13090v1) [paper-pdf](http://arxiv.org/pdf/2506.13090v1)

**Authors**: Chidera Biringa, Gokhan Kul

**Abstract**: Software developers frequently hard-code credentials such as passwords, generic secrets, private keys, and generic tokens in software repositories, even though it is strictly advised against due to the severe threat to the security of the software. These credentials create attack surfaces exploitable by a potential adversary to conduct malicious exploits such as backdoor attacks. Recent detection efforts utilize embedding models to vectorize textual credentials before passing them to classifiers for predictions. However, these models struggle to discriminate between credentials with contextual and complex sequences resulting in high false positive predictions. Context-dependent Pre-trained Language Models (PLMs) or Large Language Models (LLMs) such as Generative Pre-trained Transformers (GPT) tackled this drawback by leveraging the transformer neural architecture capacity for self-attention to capture contextual dependencies between words in input sequences. As a result, GPT has achieved wide success in several natural language understanding endeavors. Hence, we assess LLMs to represent these observations and feed extracted embedding vectors to a deep learning classifier to detect hard-coded credentials. Our model outperforms the current state-of-the-art by 13% in F1 measure on the benchmark dataset. We have made all source code and data publicly available to facilitate the reproduction of all results presented in this paper.



## **20. Detecting Adversarial Examples**

cs.LG

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2410.17442v2) [paper-pdf](http://arxiv.org/pdf/2410.17442v2)

**Authors**: Furkan Mumcu, Yasin Yilmaz

**Abstract**: Deep Neural Networks (DNNs) have been shown to be vulnerable to adversarial examples. While numerous successful adversarial attacks have been proposed, defenses against these attacks remain relatively understudied. Existing defense approaches either focus on negating the effects of perturbations caused by the attacks to restore the DNNs' original predictions or use a secondary model to detect adversarial examples. However, these methods often become ineffective due to the continuous advancements in attack techniques. We propose a novel universal and lightweight method to detect adversarial examples by analyzing the layer outputs of DNNs. Our method trains a lightweight regression model that predicts deeper-layer features from early-layer features, and uses the prediction error to detect adversarial samples. Through theoretical justification and extensive experiments, we demonstrate that our detection method is highly effective, compatible with any DNN architecture, and applicable across different domains, such as image, video, and audio.



## **21. Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories**

cs.AI

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12911v1) [paper-pdf](http://arxiv.org/pdf/2506.12911v1)

**Authors**: Pantelis Dogoulis, Fabien Bernier, Félix Fourreau, Karim Tit, Maxime Cordy

**Abstract**: Many real-world machine learning tasks require outputs that satisfy hard constraints, such as physical conservation laws, structured dependencies in graphs, or column-level relationships in tabular data. Existing approaches rely either on domain-specific architectures and losses or on strong assumptions on the constraint space, restricting their applicability to linear or convex constraints. We propose a general-purpose framework for constraint-aware refinement that leverages denoising diffusion implicit models (DDIMs). Starting from a coarse prediction, our method iteratively refines it through a deterministic diffusion trajectory guided by a learned prior and augmented by constraint gradient corrections. The approach accommodates a wide class of non-convex and nonlinear equality constraints and can be applied post hoc to any base model. We demonstrate the method in two representative domains: constrained adversarial attack generation on tabular data with column-level dependencies and in AC power flow prediction under Kirchhoff's laws. Across both settings, our diffusion-guided refinement improves both constraint satisfaction and performance while remaining lightweight and model-agnostic.



## **22. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12880v1) [paper-pdf](http://arxiv.org/pdf/2506.12880v1)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack (Zou et al., 2023), we observe that suffixes vary in efficacy: some markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that GCG's effectiveness is driven by a shallow, critical mechanism, built on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving attack success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.



## **23. Intriguing Frequency Interpretation of Adversarial Robustness for CNNs and ViTs**

cs.CV

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12875v1) [paper-pdf](http://arxiv.org/pdf/2506.12875v1)

**Authors**: Lu Chen, Han Yang, Hu Wang, Yuxin Cao, Shaofeng Li, Yuan Luo

**Abstract**: Adversarial examples have attracted significant attention over the years, yet understanding their frequency-based characteristics remains insufficient. In this paper, we investigate the intriguing properties of adversarial examples in the frequency domain for the image classification task, with the following key findings. (1) As the high-frequency components increase, the performance gap between adversarial and natural examples becomes increasingly pronounced. (2) The model performance against filtered adversarial examples initially increases to a peak and declines to its inherent robustness. (3) In Convolutional Neural Networks, mid- and high-frequency components of adversarial examples exhibit their attack capabilities, while in Transformers, low- and mid-frequency components of adversarial examples are particularly effective. These results suggest that different network architectures have different frequency preferences and that differences in frequency components between adversarial and natural examples may directly influence model robustness. Based on our findings, we further conclude with three useful proposals that serve as a valuable reference to the AI model security community.



## **24. Active Adversarial Noise Suppression for Image Forgery Localization**

cs.CV

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12871v1) [paper-pdf](http://arxiv.org/pdf/2506.12871v1)

**Authors**: Rongxuan Peng, Shunquan Tan, Xianbo Mo, Alex C. Kot, Jiwu Huang

**Abstract**: Recent advances in deep learning have significantly propelled the development of image forgery localization. However, existing models remain highly vulnerable to adversarial attacks: imperceptible noise added to forged images can severely mislead these models. In this paper, we address this challenge with an Adversarial Noise Suppression Module (ANSM) that generate a defensive perturbation to suppress the attack effect of adversarial noise. We observe that forgery-relevant features extracted from adversarial and original forged images exhibit distinct distributions. To bridge this gap, we introduce Forgery-relevant Features Alignment (FFA) as a first-stage training strategy, which reduces distributional discrepancies by minimizing the channel-wise Kullback-Leibler divergence between these features. To further refine the defensive perturbation, we design a second-stage training strategy, termed Mask-guided Refinement (MgR), which incorporates a dual-mask constraint. MgR ensures that the perturbation remains effective for both adversarial and original forged images, recovering forgery localization accuracy to their original level. Extensive experiments across various attack algorithms demonstrate that our method significantly restores the forgery localization model's performance on adversarial images. Notably, when ANSM is applied to original forged images, the performance remains nearly unaffected. To our best knowledge, this is the first report of adversarial defense in image forgery localization tasks. We have released the source code and anti-forensics dataset.



## **25. Privacy-Preserving Federated Learning against Malicious Clients Based on Verifiable Functional Encryption**

cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12846v1) [paper-pdf](http://arxiv.org/pdf/2506.12846v1)

**Authors**: Nina Cai, Jinguang Han

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods.



## **26. Transforming Chatbot Text: A Sequence-to-Sequence Approach**

cs.CL

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12843v1) [paper-pdf](http://arxiv.org/pdf/2506.12843v1)

**Authors**: Natesh Reddy, Mark Stamp

**Abstract**: Due to advances in Large Language Models (LLMs) such as ChatGPT, the boundary between human-written text and AI-generated text has become blurred. Nevertheless, recent work has demonstrated that it is possible to reliably detect GPT-generated text. In this paper, we adopt a novel strategy to adversarially transform GPT-generated text using sequence-to-sequence (Seq2Seq) models, with the goal of making the text more human-like. We experiment with the Seq2Seq models T5-small and BART which serve to modify GPT-generated sentences to include linguistic, structural, and semantic components that may be more typical of human-authored text. Experiments show that classification models trained to distinguish GPT-generated text are significantly less accurate when tested on text that has been modified by these Seq2Seq models. However, after retraining classification models on data generated by our Seq2Seq technique, the models are able to distinguish the transformed GPT-generated text from human-generated text with high accuracy. This work adds to the accumulating knowledge of text transformation as a tool for both attack -- in the sense of defeating classification models -- and defense -- in the sense of improved classifiers -- thereby advancing our understanding of AI-generated text.



## **27. I Know What You Said: Unveiling Hardware Cache Side-Channels in Local Large Language Model Inference**

cs.CR

Submitted for review in January 22, 2025, revised under shepherding

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2505.06738v3) [paper-pdf](http://arxiv.org/pdf/2505.06738v3)

**Authors**: Zibo Gao, Junjie Hu, Feng Guo, Yixin Zhang, Yinglong Han, Siyuan Liu, Haiyang Li, Zhiqiang Lv

**Abstract**: Large Language Models (LLMs) that can be deployed locally have recently gained popularity for privacy-sensitive tasks, with companies such as Meta, Google, and Intel playing significant roles in their development. However, the security of local LLMs through the lens of hardware cache side-channels remains unexplored. In this paper, we unveil novel side-channel vulnerabilities in local LLM inference: token value and token position leakage, which can expose both the victim's input and output text, thereby compromising user privacy. Specifically, we found that adversaries can infer the token values from the cache access patterns of the token embedding operation, and deduce the token positions from the timing of autoregressive decoding phases. To demonstrate the potential of these leaks, we design a novel eavesdropping attack framework targeting both open-source and proprietary LLM inference systems. The attack framework does not directly interact with the victim's LLM and can be executed without privilege.   We evaluate the attack on a range of practical local LLM deployments (e.g., Llama, Falcon, and Gemma), and the results show that our attack achieves promising accuracy. The restored output and input text have an average edit distance of 5.2% and 17.3% to the ground truth, respectively. Furthermore, the reconstructed texts achieve average cosine similarity scores of 98.7% (input) and 98.0% (output).



## **28. SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression**

cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12707v1) [paper-pdf](http://arxiv.org/pdf/2506.12707v1)

**Authors**: Yucheng Li, Surin Ahn, Huiqiang Jiang, Amir H. Abdi, Yuqing Yang, Lili Qiu

**Abstract**: Large language models (LLMs) have achieved widespread adoption across numerous applications. However, many LLMs are vulnerable to malicious attacks even after safety alignment. These attacks typically bypass LLMs' safety guardrails by wrapping the original malicious instructions inside adversarial jailbreaks prompts. Previous research has proposed methods such as adversarial training and prompt rephrasing to mitigate these safety vulnerabilities, but these methods often reduce the utility of LLMs or lead to significant computational overhead and online latency. In this paper, we propose SecurityLingua, an effective and efficient approach to defend LLMs against jailbreak attacks via security-oriented prompt compression. Specifically, we train a prompt compressor designed to discern the "true intention" of the input prompt, with a particular focus on detecting the malicious intentions of adversarial prompts. Then, in addition to the original prompt, the intention is passed via the system prompt to the target LLM to help it identify the true intention of the request. SecurityLingua ensures a consistent user experience by leaving the original input prompt intact while revealing the user's potentially malicious intention and stimulating the built-in safety guardrails of the LLM. Moreover, thanks to prompt compression, SecurityLingua incurs only a negligible overhead and extra token cost compared to all existing defense methods, making it an especially practical solution for LLM defense. Experimental results demonstrate that SecurityLingua can effectively defend against malicious attacks and maintain utility of the LLM with negligible compute and latency overhead. Our code is available at https://aka.ms/SecurityLingua.



## **29. NAP-Tuning: Neural Augmented Prompt Tuning for Adversarially Robust Vision-Language Models**

cs.CV

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12706v1) [paper-pdf](http://arxiv.org/pdf/2506.12706v1)

**Authors**: Jiaming Zhang, Xin Wang, Xingjun Ma, Lingyu Qiu, Yu-Gang Jiang, Jitao Sang

**Abstract**: Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capabilities in understanding relationships between visual and textual data through joint embedding spaces. Despite their effectiveness, these models remain vulnerable to adversarial attacks, particularly in the image modality, posing significant security concerns. Building upon our previous work on Adversarial Prompt Tuning (AdvPT), which introduced learnable text prompts to enhance adversarial robustness in VLMs without extensive parameter training, we present a significant extension by introducing the Neural Augmentor framework for Multi-modal Adversarial Prompt Tuning (NAP-Tuning).Our key innovations include: (1) extending AdvPT from text-only to multi-modal prompting across both text and visual modalities, (2) expanding from single-layer to multi-layer prompt architectures, and (3) proposing a novel architecture-level redesign through our Neural Augmentor approach, which implements feature purification to directly address the distortions introduced by adversarial attacks in feature space. Our NAP-Tuning approach incorporates token refiners that learn to reconstruct purified features through residual connections, allowing for modality-specific and layer-specific feature correction.Comprehensive experiments demonstrate that NAP-Tuning significantly outperforms existing methods across various datasets and attack types. Notably, our approach shows significant improvements over the strongest baselines under the challenging AutoAttack benchmark, outperforming them by 33.5% on ViT-B16 and 33.0% on ViT-B32 architectures while maintaining competitive clean accuracy.



## **30. Zeroth-Order primal-dual Alternating Projection Gradient Algorithms for Nonconvex Minimax Problems with Coupled linear Constraints**

math.OC

arXiv admin note: text overlap with arXiv:2212.04672

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2402.03352v2) [paper-pdf](http://arxiv.org/pdf/2402.03352v2)

**Authors**: Huiling Zhang, Zi Xu, Yuhong Dai

**Abstract**: In this paper, we study zeroth-order algorithms for nonconvex minimax problems with coupled linear constraints under the deterministic and stochastic settings, which have attracted wide attention in machine learning, signal processing and many other fields in recent years, e.g., adversarial attacks in resource allocation problems and network flow problems etc. We propose two single-loop algorithms, namely the zeroth-order primal-dual alternating projected gradient (ZO-PDAPG) algorithm and the zeroth-order regularized momentum primal-dual projected gradient algorithm (ZO-RMPDPG), for solving deterministic and stochastic nonconvex-(strongly) concave minimax problems with coupled linear constraints. The iteration complexity of the two proposed algorithms to obtain an $\varepsilon$-stationary point are proved to be $\mathcal{O}(\varepsilon ^{-2})$ (resp. $\mathcal{O}(\varepsilon ^{-4})$) for solving nonconvex-strongly concave (resp. nonconvex-concave) minimax problems with coupled linear constraints under deterministic settings and $\tilde{\mathcal{O}}(\varepsilon ^{-3})$ (resp. $\tilde{\mathcal{O}}(\varepsilon ^{-6.5})$) under stochastic settings respectively. To the best of our knowledge, they are the first two zeroth-order algorithms with iterative complexity guarantees for solving nonconvex-(strongly) concave minimax problems with coupled linear constraints under the deterministic and stochastic settings.



## **31. Alphabet Index Mapping: Jailbreaking LLMs through Semantic Dissimilarity**

cs.CR

10 pages, 2 figures, 3 tables

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12685v1) [paper-pdf](http://arxiv.org/pdf/2506.12685v1)

**Authors**: Bilal Saleh Husain

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their susceptibility to adversarial attacks, particularly jailbreaking, poses significant safety and ethical concerns. While numerous jailbreak methods exist, many suffer from computational expense, high token usage, or complex decoding schemes. Liu et al. (2024) introduced FlipAttack, a black-box method that achieves high attack success rates (ASR) through simple prompt manipulation. This paper investigates the underlying mechanisms of FlipAttack's effectiveness by analyzing the semantic changes induced by its flipping modes. We hypothesize that semantic dissimilarity between original and manipulated prompts is inversely correlated with ASR. To test this, we examine embedding space visualizations (UMAP, KDE) and cosine similarities for FlipAttack's modes. Furthermore, we introduce a novel adversarial attack, Alphabet Index Mapping (AIM), designed to maximize semantic dissimilarity while maintaining simple decodability. Experiments on GPT-4 using a subset of AdvBench show AIM and its variant AIM+FWO achieve a 94% ASR, outperforming FlipAttack and other methods on this subset. Our findings suggest that while high semantic dissimilarity is crucial, a balance with decoding simplicity is key for successful jailbreaking. This work contributes to a deeper understanding of adversarial prompt mechanics and offers a new, effective jailbreak technique.



## **32. FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training**

cs.LG

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2410.23142v3) [paper-pdf](http://arxiv.org/pdf/2410.23142v3)

**Authors**: Tejaswini Medi, Steffen Jung, Margret Keuper

**Abstract**: Deep neural networks are susceptible to adversarial attacks and common corruptions, which undermine their robustness. In order to enhance model resilience against such challenges, Adversarial Training (AT) has emerged as a prominent solution. Nevertheless, adversarial robustness is often attained at the expense of model fairness during AT, i.e., disparity in class-wise robustness of the model. While distinctive classes become more robust towards such adversaries, hard to detect classes suffer. Recently, research has focused on improving model fairness specifically for perturbed images, overlooking the accuracy of the most likely non-perturbed data. Additionally, despite their robustness against the adversaries encountered during model training, state-of-the-art adversarial trained models have difficulty maintaining robustness and fairness when confronted with diverse adversarial threats or common corruptions. In this work, we address the above concerns by introducing a novel approach called Fair Targeted Adversarial Training (FAIR-TAT). We show that using targeted adversarial attacks for adversarial training (instead of untargeted attacks) can allow for more favorable trade-offs with respect to adversarial fairness. Empirical results validate the efficacy of our approach.



## **33. Control Tax: The Price of Keeping AI in Check**

cs.AI

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.05296v2) [paper-pdf](http://arxiv.org/pdf/2506.05296v2)

**Authors**: Mikhail Terekhov, Zhen Ning David Liu, Caglar Gulcehre, Samuel Albanie

**Abstract**: The rapid integration of agentic AI into high-stakes real-world applications requires robust oversight mechanisms. The emerging field of AI Control (AIC) aims to provide such an oversight mechanism, but practical adoption depends heavily on implementation overhead. To study this problem better, we introduce the notion of Control tax -- the operational and financial cost of integrating control measures into AI pipelines. Our work makes three key contributions to the field of AIC: (1) we introduce a theoretical framework that quantifies the Control Tax and maps classifier performance to safety assurances; (2) we conduct comprehensive evaluations of state-of-the-art language models in adversarial settings, where attacker models insert subtle backdoors into code while monitoring models attempt to detect these vulnerabilities; and (3) we provide empirical financial cost estimates for control protocols and develop optimized monitoring strategies that balance safety and cost-effectiveness while accounting for practical constraints like auditing budgets. Our framework enables practitioners to make informed decisions by systematically connecting safety guarantees with their costs, advancing AIC through principled economic feasibility assessment across different deployment contexts.



## **34. Exploiting AI for Attacks: On the Interplay between Adversarial AI and Offensive AI**

cs.CR

under submission

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12519v1) [paper-pdf](http://arxiv.org/pdf/2506.12519v1)

**Authors**: Saskia Laura Schröer, Luca Pajola, Alberto Castagnaro, Giovanni Apruzzese, Mauro Conti

**Abstract**: As Artificial Intelligence (AI) continues to evolve, it has transitioned from a research-focused discipline to a widely adopted technology, enabling intelligent solutions across various sectors. In security, AI's role in strengthening organizational resilience has been studied for over two decades. While much attention has focused on AI's constructive applications, the increasing maturity and integration of AI have also exposed its darker potentials. This article explores two emerging AI-related threats and the interplay between them: AI as a target of attacks (`Adversarial AI') and AI as a means to launch attacks on any target (`Offensive AI') -- potentially even on another AI. By cutting through the confusion and explaining these threats in plain terms, we introduce the complex and often misunderstood interplay between Adversarial AI and Offensive AI, offering a clear and accessible introduction to the challenges posed by these threats.



## **35. On the existence of consistent adversarial attacks in high-dimensional linear classification**

stat.ML

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12454v1) [paper-pdf](http://arxiv.org/pdf/2506.12454v1)

**Authors**: Matteo Vilucchio, Lenka Zdeborová, Bruno Loureiro

**Abstract**: What fundamentally distinguishes an adversarial attack from a misclassification due to limited model expressivity or finite data? In this work, we investigate this question in the setting of high-dimensional binary classification, where statistical effects due to limited data availability play a central role. We introduce a new error metric that precisely capture this distinction, quantifying model vulnerability to consistent adversarial attacks -- perturbations that preserve the ground-truth labels. Our main technical contribution is an exact and rigorous asymptotic characterization of these metrics in both well-specified models and latent space models, revealing different vulnerability patterns compared to standard robust error measures. The theoretical results demonstrate that as models become more overparameterized, their vulnerability to label-preserving perturbations grows, offering theoretical insight into the mechanisms underlying model sensitivity to adversarial attacks.



## **36. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

cs.CR

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12430v1) [paper-pdf](http://arxiv.org/pdf/2506.12430v1)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.



## **37. InverTune: Removing Backdoors from Multimodal Contrastive Learning Models via Trigger Inversion and Activation Tuning**

cs.CR

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12411v1) [paper-pdf](http://arxiv.org/pdf/2506.12411v1)

**Authors**: Mengyuan Sun, Yu Li, Yuchen Liu, Bo Du, Yunjie Ge

**Abstract**: Multimodal contrastive learning models like CLIP have demonstrated remarkable vision-language alignment capabilities, yet their vulnerability to backdoor attacks poses critical security risks. Attackers can implant latent triggers that persist through downstream tasks, enabling malicious control of model behavior upon trigger presentation. Despite great success in recent defense mechanisms, they remain impractical due to strong assumptions about attacker knowledge or excessive clean data requirements. In this paper, we introduce InverTune, the first backdoor defense framework for multimodal models under minimal attacker assumptions, requiring neither prior knowledge of attack targets nor access to the poisoned dataset. Unlike existing defense methods that rely on the same dataset used in the poisoning stage, InverTune effectively identifies and removes backdoor artifacts through three key components, achieving robust protection against backdoor attacks. Specifically, InverTune first exposes attack signatures through adversarial simulation, probabilistically identifying the target label by analyzing model response patterns. Building on this, we develop a gradient inversion technique to reconstruct latent triggers through activation pattern analysis. Finally, a clustering-guided fine-tuning strategy is employed to erase the backdoor function with only a small amount of arbitrary clean data, while preserving the original model capabilities. Experimental results show that InverTune reduces the average attack success rate (ASR) by 97.87% against the state-of-the-art (SOTA) attacks while limiting clean accuracy (CA) degradation to just 3.07%. This work establishes a new paradigm for securing multimodal systems, advancing security in foundation model deployment without compromising performance.



## **38. Exploring the Secondary Risks of Large Language Models**

cs.LG

18 pages, 5 figures

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12382v1) [paper-pdf](http://arxiv.org/pdf/2506.12382v1)

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.



## **39. InfoFlood: Jailbreaking Large Language Models with Information Overload**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.12274v1) [paper-pdf](http://arxiv.org/pdf/2506.12274v1)

**Authors**: Advait Yadav, Haibo Jin, Man Luo, Jun Zhuang, Haohan Wang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. However, their potential to generate harmful responses has raised significant societal and regulatory concerns, especially when manipulated by adversarial techniques known as "jailbreak" attacks. Existing jailbreak methods typically involve appending carefully crafted prefixes or suffixes to malicious prompts in order to bypass the built-in safety mechanisms of these models.   In this work, we identify a new vulnerability in which excessive linguistic complexity can disrupt built-in safety mechanisms-without the need for any added prefixes or suffixes-allowing attackers to elicit harmful outputs directly. We refer to this phenomenon as Information Overload.   To automatically exploit this vulnerability, we propose InfoFlood, a jailbreak attack that transforms malicious queries into complex, information-overloaded queries capable of bypassing built-in safety mechanisms. Specifically, InfoFlood: (1) uses linguistic transformations to rephrase malicious queries, (2) identifies the root cause of failure when an attempt is unsuccessful, and (3) refines the prompt's linguistic structure to address the failure while preserving its malicious intent.   We empirically validate the effectiveness of InfoFlood on four widely used LLMs-GPT-4o, GPT-3.5-turbo, Gemini 2.0, and LLaMA 3.1-by measuring their jailbreak success rates. InfoFlood consistently outperforms baseline attacks, achieving up to 3 times higher success rates across multiple jailbreak benchmarks. Furthermore, we demonstrate that commonly adopted post-processing defenses, including OpenAI's Moderation API, Perspective API, and SmoothLLM, fail to mitigate these attacks. This highlights a critical weakness in traditional AI safety guardrails when confronted with information overload-based jailbreaks.



## **40. AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security**

cs.LG

ICLR 2025 Workshop BuildingTrust

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2504.20965v2) [paper-pdf](http://arxiv.org/pdf/2504.20965v2)

**Authors**: Zikui Cai, Shayan Shabihi, Bang An, Zora Che, Brian R. Bartoldson, Bhavya Kailkhura, Tom Goldstein, Furong Huang

**Abstract**: We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm



## **41. Learning Causality for Modern Machine Learning**

cs.LG

PhD thesis

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.12226v1) [paper-pdf](http://arxiv.org/pdf/2506.12226v1)

**Authors**: Yongqiang Chen

**Abstract**: In the past decades, machine learning with Empirical Risk Minimization (ERM) has demonstrated great capability in learning and exploiting the statistical patterns from data, or even surpassing humans. Despite the success, ERM avoids the modeling of causality the way of understanding and handling changes, which is fundamental to human intelligence. When deploying models beyond the training environment, distribution shifts are everywhere. For example, an autopilot system often needs to deal with new weather conditions that have not been seen during training, An Al-aided drug discovery system needs to predict the biochemical properties of molecules with respect to new viruses such as COVID-19. It renders the problem of Out-of-Distribution (OOD) generalization challenging to conventional machine learning.   In this thesis, we investigate how to incorporate and realize the causality for broader tasks in modern machine learning. In particular, we exploit the invariance implied by the principle of independent causal mechanisms (ICM), that is, the causal mechanisms generating the effects from causes do not inform or influence each other. Therefore, the conditional distribution between the target variable given its causes is invariant under distribution shifts. With the causal invariance principle, we first instantiate it to graphs -- a general data structure ubiquitous in many real-world industry and scientific applications, such as financial networks and molecules. Then, we shall see how learning the causality benefits many of the desirable properties of modern machine learning, in terms of (i) OOD generalization capability; (ii) interpretability; and (iii) robustness to adversarial attacks.   Realizing the causality in machine learning, on the other hand, raises a dilemma for optimization in conventional machine learning, as it often contradicts the objective of ERM...



## **42. Self-interpreting Adversarial Images**

cs.CR

in USENIX Security 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2407.08970v4) [paper-pdf](http://arxiv.org/pdf/2407.08970v4)

**Authors**: Tingwei Zhang, Collin Zhang, John X. Morris, Eugene Bagdasarian, Vitaly Shmatikov

**Abstract**: We introduce a new type of indirect, cross-modal injection attacks against visual language models that enable creation of self-interpreting images. These images contain hidden "meta-instructions" that control how models answer users' questions about the image and steer models' outputs to express an adversary-chosen style, sentiment, or point of view.   Self-interpreting images act as soft prompts, conditioning the model to satisfy the adversary's (meta-)objective while still producing answers based on the image's visual content. Meta-instructions are thus a stronger form of prompt injection. Adversarial images look natural and the model's answers are coherent and plausible, yet they also follow the adversary-chosen interpretation, e.g., political spin, or even objectives that are not achievable with explicit text instructions.   We evaluate the efficacy of self-interpreting images for a variety of models, interpretations, and user prompts. We describe how these attacks could cause harm by enabling creation of self-interpreting content that carries spam, misinformation, or spin. Finally, we discuss defenses.



## **43. Improving Large Language Model Safety with Contrastive Representation Learning**

cs.CL

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11938v1) [paper-pdf](http://arxiv.org/pdf/2506.11938v1)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense



## **44. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2501.18638v2) [paper-pdf](http://arxiv.org/pdf/2501.18638v2)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.



## **45. Attention-based Adversarial Robust Distillation in Radio Signal Classifications for Low-Power IoT Devices**

cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11892v1) [paper-pdf](http://arxiv.org/pdf/2506.11892v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Basil AsSadhan, Fabio Roli

**Abstract**: Due to great success of transformers in many applications such as natural language processing and computer vision, transformers have been successfully applied in automatic modulation classification. We have shown that transformer-based radio signal classification is vulnerable to imperceptible and carefully crafted attacks called adversarial examples. Therefore, we propose a defense system against adversarial examples in transformer-based modulation classifications. Considering the need for computationally efficient architecture particularly for Internet of Things (IoT)-based applications or operation of devices in environment where power supply is limited, we propose a compact transformer for modulation classification. The advantages of robust training such as adversarial training in transformers may not be attainable in compact transformers. By demonstrating this, we propose a novel compact transformer that can enhance robustness in the presence of adversarial attacks. The new method is aimed at transferring the adversarial attention map from the robustly trained large transformer to a compact transformer. The proposed method outperforms the state-of-the-art techniques for the considered white-box scenarios including fast gradient method and projected gradient descent attacks. We have provided reasoning of the underlying working mechanisms and investigated the transferability of the adversarial examples between different architectures. The proposed method has the potential to protect the transformer from the transferability of adversarial examples.



## **46. Black-Box Adversarial Attacks on LLM-Based Code Completion**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2408.02509v2) [paper-pdf](http://arxiv.org/pdf/2408.02509v2)

**Authors**: Slobodan Jenko, Niels Mündler, Jingxuan He, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.



## **47. TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks**

cs.LG

12 pages, 5 figures, in KDD 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11844v1) [paper-pdf](http://arxiv.org/pdf/2506.11844v1)

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field.



## **48. Differential Privacy in Machine Learning: From Symbolic AI to LLMs**

cs.CR

arXiv admin note: text overlap with arXiv:2303.00654 by other authors

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11687v1) [paper-pdf](http://arxiv.org/pdf/2506.11687v1)

**Authors**: Francisco Aguilera-Martínez, Fernando Berzal

**Abstract**: Machine learning models should not reveal particular information that is not otherwise accessible. Differential privacy provides a formal framework to mitigate privacy risks by ensuring that the inclusion or exclusion of any single data point does not significantly alter the output of an algorithm, thus limiting the exposure of private information. This survey paper explores the foundational definitions of differential privacy, reviews its original formulations and tracing its evolution through key research contributions. It then provides an in-depth examination of how DP has been integrated into machine learning models, analyzing existing proposals and methods to preserve privacy when training ML models. Finally, it describes how DP-based ML techniques can be evaluated in practice. %Finally, it discusses the broader implications of DP, highlighting its potential for public benefit, its real-world applications, and the challenges it faces, including vulnerabilities to adversarial attacks. By offering a comprehensive overview of differential privacy in machine learning, this work aims to contribute to the ongoing development of secure and responsible AI systems.



## **49. KCES: Training-Free Defense for Robust Graph Neural Networks via Kernel Complexity**

cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11611v1) [paper-pdf](http://arxiv.org/pdf/2506.11611v1)

**Authors**: Yaning Jia, Shenyang Deng, Chiyu Ma, Yaoqing Yang, Soroush Vosoughi

**Abstract**: Graph Neural Networks (GNNs) have achieved impressive success across a wide range of graph-based tasks, yet they remain highly vulnerable to small, imperceptible perturbations and adversarial attacks. Although numerous defense methods have been proposed to address these vulnerabilities, many rely on heuristic metrics, overfit to specific attack patterns, and suffer from high computational complexity. In this paper, we propose Kernel Complexity-Based Edge Sanitization (KCES), a training-free, model-agnostic defense framework. KCES leverages Graph Kernel Complexity (GKC), a novel metric derived from the graph's Gram matrix that characterizes GNN generalization via its test error bound. Building on GKC, we define a KC score for each edge, measuring the change in GKC when the edge is removed. Edges with high KC scores, typically introduced by adversarial perturbations, are pruned to mitigate their harmful effects, thereby enhancing GNNs' robustness. KCES can also be seamlessly integrated with existing defense strategies as a plug-and-play module without requiring training. Theoretical analysis and extensive experiments demonstrate that KCES consistently enhances GNN robustness, outperforms state-of-the-art baselines, and amplifies the effectiveness of existing defenses, offering a principled and efficient solution for securing GNNs.



## **50. Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11521v1) [paper-pdf](http://arxiv.org/pdf/2506.11521v1)

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense.



