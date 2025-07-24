# Latest Adversarial Attack Papers
**update at 2025-07-24 10:59:36**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Constructing Optimal Noise Channels for Enhanced Robustness in Quantum Machine Learning**

quant-ph

QML technical track at IEEE QCE 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2404.16417v2) [paper-pdf](http://arxiv.org/pdf/2404.16417v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: With the rapid advancement of Quantum Machine Learning (QML), the critical need to enhance security measures against adversarial attacks and protect QML models becomes increasingly evident. In this work, we outline the connection between quantum noise channels and differential privacy (DP), by constructing a family of noise channels which are inherently $\epsilon$-DP: $(\alpha, \gamma)$-channels. Through this approach, we successfully replicate the $\epsilon$-DP bounds observed for depolarizing and random rotation channels, thereby affirming the broad generality of our framework. Additionally, we use a semi-definite program to construct an optimally robust channel. In a small-scale experimental evaluation, we demonstrate the benefits of using our optimal noise channel over depolarizing noise, particularly in enhancing adversarial accuracy. Moreover, we assess how the variables $\alpha$ and $\gamma$ affect the certifiable robustness and investigate how different encoding methods impact the classifier's robustness.



## **2. Boosting Ray Search Procedure of Hard-label Attacks with Transfer-based Priors**

cs.CV

Published at ICLR 2025 (Spotlight paper)

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17577v1) [paper-pdf](http://arxiv.org/pdf/2507.17577v1)

**Authors**: Chen Ma, Xinjie Xu, Shuyu Cheng, Qi Xuan

**Abstract**: One of the most practical and challenging types of black-box adversarial attacks is the hard-label attack, where only the top-1 predicted label is available. One effective approach is to search for the optimal ray direction from the benign image that minimizes the $\ell_p$-norm distance to the adversarial region. The unique advantage of this approach is that it transforms the hard-label attack into a continuous optimization problem. The objective function value is the ray's radius, which can be obtained via binary search at a high query cost. Existing methods use a "sign trick" in gradient estimation to reduce the number of queries. In this paper, we theoretically analyze the quality of this gradient estimation and propose a novel prior-guided approach to improve ray search efficiency both theoretically and empirically. Specifically, we utilize the transfer-based priors from surrogate models, and our gradient estimators appropriately integrate them by approximating the projection of the true gradient onto the subspace spanned by these priors and random directions, in a query-efficient manner. We theoretically derive the expected cosine similarities between the obtained gradient estimators and the true gradient, and demonstrate the improvement achieved by incorporating priors. Extensive experiments on the ImageNet and CIFAR-10 datasets show that our approach significantly outperforms 11 state-of-the-art methods in terms of query efficiency.



## **3. An h-space Based Adversarial Attack for Protection Against Few-shot Personalization**

cs.CV

32 pages, 15 figures. Accepted by ACM Multimedia 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17554v1) [paper-pdf](http://arxiv.org/pdf/2507.17554v1)

**Authors**: Xide Xu, Sandesh Kamath, Muhammad Atif Butt, Bogdan Raducanu

**Abstract**: The versatility of diffusion models in generating customized images from few samples raises significant privacy concerns, particularly regarding unauthorized modifications of private content. This concerning issue has renewed the efforts in developing protection mechanisms based on adversarial attacks, which generate effective perturbations to poison diffusion models. Our work is motivated by the observation that these models exhibit a high degree of abstraction within their semantic latent space (`h-space'), which encodes critical high-level features for generating coherent and meaningful content. In this paper, we propose a novel anti-customization approach, called HAAD (h-space based Adversarial Attack for Diffusion models), that leverages adversarial attacks to craft perturbations based on the h-space that can efficiently degrade the image generation process. Building upon HAAD, we further introduce a more efficient variant, HAAD-KV, that constructs perturbations solely based on the KV parameters of the h-space. This strategy offers a stronger protection, that is computationally less expensive. Despite their simplicity, our methods outperform state-of-the-art adversarial attacks, highlighting their effectiveness.



## **4. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2505.23404v4) [paper-pdf](http://arxiv.org/pdf/2505.23404v4)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Based on our experiments, we found that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the attacked LLM. Building on this insight, we propose a capability-aware Multi-Encryption Framework (MEF) for evaluating vulnerabilities in black-box LLMs. Specifically, MEF first categorizes the comprehension ability level of the LLM, then applies different strategies accordingly: For models with limited comprehension ability, MEF adopts the Fu+En1 strategy, which integrates layered semantic mutations with an encryption technique, more effectively contributing to evasion of the LLM's defenses at the input and inference stages. For models with strong comprehension ability, MEF uses a more complex Fu+En1+En2 strategy, in which additional dual-ended encryption techniques are applied to the LLM's responses, further contributing to evasion of the LLM's defenses at the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of 98.9% on GPT-4o (29 May 2025 release) and 99.8% on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLM alignment mechanisms.



## **5. Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks**

cs.SE

Accepted to ICSME 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.10054v2) [paper-pdf](http://arxiv.org/pdf/2507.10054v2)

**Authors**: Emir Bosnak, Sahand Moslemi, Mayasah Lami, Anil Koyuncu

**Abstract**: Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities, this study examines a more direct threat: open-source LLMs generating vulnerable code when prompted. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and prompt phrasing across structured templates; and (2) Reverse Prompting, which derives natural-language prompts from real vulnerable code samples. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, Gemma) using static analysis to assess both the presence and correctness of generated vulnerabilities. Our results show that all models frequently generate the requested vulnerabilities, though with significant performance differences. Gemma achieves the highest correctness for memory vulnerabilities under Dynamic Prompting (e.g., 98.6% for buffer overflows), while Qwen2 demonstrates the most balanced performance across all tasks. We find that professional personas (e.g., "DevOps Engineer") consistently elicit higher success rates than student personas, and that the effectiveness of direct versus indirect phrasing is inverted depending on the prompting strategy. Vulnerability reproduction accuracy follows a non-linear pattern with code complexity, peaking in a moderate range. Our findings expose how LLMs' reliance on pattern recall over semantic reasoning creates significant blind spots in their safety alignments, particularly for requests framed as plausible professional tasks.



## **6. Optimizing Privacy-Utility Trade-off in Decentralized Learning with Generalized Correlated Noise**

cs.LG

6 pages, 5 figures, accepted at IEEE ITW 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2501.14644v2) [paper-pdf](http://arxiv.org/pdf/2501.14644v2)

**Authors**: Angelo Rodio, Zheng Chen, Erik G. Larsson

**Abstract**: Decentralized learning enables distributed agents to collaboratively train a shared machine learning model without a central server, through local computation and peer-to-peer communication. Although each agent retains its dataset locally, sharing local models can still expose private information about the local training datasets to adversaries. To mitigate privacy attacks, a common strategy is to inject random artificial noise at each agent before exchanging local models between neighbors. However, this often leads to utility degradation due to the negative effects of cumulated artificial noise on the learning algorithm. In this work, we introduce CorN-DSGD, a novel covariance-based framework for generating correlated privacy noise across agents, which unifies several state-of-the-art methods as special cases. By leveraging network topology and mixing weights, CorN-DSGD optimizes the noise covariance to achieve network-wide noise cancellation. Experimental results show that CorN-DSGD cancels more noise than existing pairwise correlation schemes, improving model performance under formal privacy guarantees.



## **7. Restricted Boltzmann machine as a probabilistic Enigma**

cond-mat.stat-mech

7 pages, 4 figures

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17236v1) [paper-pdf](http://arxiv.org/pdf/2507.17236v1)

**Authors**: Bin Chen, Weichao Yu

**Abstract**: We theoretically propose a symmetric encryption scheme based on Restricted Boltzmann Machines that functions as a probabilistic Enigma device, encoding information in the marginal distributions of visible states while utilizing bias permutations as cryptographic keys. Theoretical analysis reveals significant advantages including factorial key space growth through permutation matrices, excellent diffusion properties, and computational complexity rooted in sharp P-complete problems that resist quantum attacks. Compatible with emerging probabilistic computing hardware, the scheme establishes an asymmetric computational barrier where legitimate users decrypt efficiently while adversaries face exponential costs. This framework unlocks probabilistic computers' potential for cryptographic systems, offering an emerging encryption paradigm between classical and quantum regimes for post-quantum security.



## **8. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

cs.CV

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2502.20650v4) [paper-pdf](http://arxiv.org/pdf/2502.20650v4)

**Authors**: Yu Pan, Jiahao Chen, Bingrong Dai, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.



## **9. Advancing Robustness in Deep Reinforcement Learning with an Ensemble Defense Approach**

cs.LG

6 pages, 4 figures, 2 tables

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.17070v1) [paper-pdf](http://arxiv.org/pdf/2507.17070v1)

**Authors**: Adithya Mohan, Dominik Rößle, Daniel Cremers, Torsten Schön

**Abstract**: Recent advancements in Deep Reinforcement Learning (DRL) have demonstrated its applicability across various domains, including robotics, healthcare, energy optimization, and autonomous driving. However, a critical question remains: How robust are DRL models when exposed to adversarial attacks? While existing defense mechanisms such as adversarial training and distillation enhance the resilience of DRL models, there remains a significant research gap regarding the integration of multiple defenses in autonomous driving scenarios specifically. This paper addresses this gap by proposing a novel ensemble-based defense architecture to mitigate adversarial attacks in autonomous driving. Our evaluation demonstrates that the proposed architecture significantly enhances the robustness of DRL models. Compared to the baseline under FGSM attacks, our ensemble method improves the mean reward from 5.87 to 18.38 (over 213% increase) and reduces the mean collision rate from 0.50 to 0.09 (an 82% decrease) in the highway scenario and merge scenario, outperforming all standalone defense strategies.



## **10. When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16773v1) [paper-pdf](http://arxiv.org/pdf/2507.16773v1)

**Authors**: Yue Li, Xiao Li, Hao Wu, Yue Zhang, Fengyuan Xu, Xiuzhen Cheng, Sheng Zhong

**Abstract**: Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.



## **11. ShadowCode: Towards (Automatic) External Prompt Injection Attack against Code LLMs**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2407.09164v6) [paper-pdf](http://arxiv.org/pdf/2407.09164v6)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Bingrun Yang, Yiling He, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: Recent advancements have led to the widespread adoption of code-oriented large language models (Code LLMs) for programming tasks. Despite their success in deployment, their security research is left far behind. This paper introduces a new attack paradigm: (automatic) external prompt injection against Code LLMs, where attackers generate concise, non-functional induced perturbations and inject them within a victim's code context. These induced perturbations can be disseminated through commonly used dependencies (e.g., packages or RAG's knowledge base), manipulating Code LLMs to achieve malicious objectives during the code completion process. Compared to existing attacks, this method is more realistic and threatening: it does not necessitate control over the model's training process, unlike backdoor attacks, and can achieve specific malicious objectives that are challenging for adversarial attacks. Furthermore, we propose ShadowCode, a simple yet effective method that automatically generates induced perturbations based on code simulation to achieve effective and stealthy external prompt injection. ShadowCode designs its perturbation optimization objectives by simulating realistic code contexts and employs a greedy optimization approach with two enhancement modules: forward reasoning enhancement and keyword-based perturbation design. We evaluate our method across 13 distinct malicious objectives, generating 31 threat cases spanning three popular programming languages. Our results demonstrate that ShadowCode successfully attacks three representative open-source Code LLMs (achieving up to a 97.9% attack success rate) and two mainstream commercial Code LLM-integrated applications (with over 90% attack success rate) across all threat cases, using only a 12-token non-functional induced perturbation. The code is available at https://github.com/LianPing-cyber/ShadowCodeEPI.



## **12. The Cost of Compression: Tight Quadratic Black-Box Attacks on Sketches for $\ell_2$ Norm Estimation**

cs.LG

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16345v1) [paper-pdf](http://arxiv.org/pdf/2507.16345v1)

**Authors**: Sara Ahmadian, Edith Cohen, Uri Stemmer

**Abstract**: Dimensionality reduction via linear sketching is a powerful and widely used technique, but it is known to be vulnerable to adversarial inputs. We study the black-box adversarial setting, where a fixed, hidden sketching matrix A in $R^{k X n}$ maps high-dimensional vectors v $\in R^n$ to lower-dimensional sketches A v in $R^k$, and an adversary can query the system to obtain approximate ell2-norm estimates that are computed from the sketch.   We present a universal, nonadaptive attack that, using tilde(O)($k^2$) queries, either causes a failure in norm estimation or constructs an adversarial input on which the optimal estimator for the query distribution (used by the attack) fails. The attack is completely agnostic to the sketching matrix and to the estimator: It applies to any linear sketch and any query responder, including those that are randomized, adaptive, or tailored to the query distribution.   Our lower bound construction tightly matches the known upper bounds of tilde(Omega)($k^2$), achieved by specialized estimators for Johnson Lindenstrauss transforms and AMS sketches. Beyond sketching, our results uncover structural parallels to adversarial attacks in image classification, highlighting fundamental vulnerabilities of compressed representations.



## **13. Talking Like a Phisher: LLM-Based Attacks on Voice Phishing Classifiers**

cs.CR

Accepted by EAI ICDF2C 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16291v1) [paper-pdf](http://arxiv.org/pdf/2507.16291v1)

**Authors**: Wenhao Li, Selvakumar Manickam, Yung-wey Chong, Shankar Karuppayah

**Abstract**: Voice phishing (vishing) remains a persistent threat in cybersecurity, exploiting human trust through persuasive speech. While machine learning (ML)-based classifiers have shown promise in detecting malicious call transcripts, they remain vulnerable to adversarial manipulations that preserve semantic content. In this study, we explore a novel attack vector where large language models (LLMs) are leveraged to generate adversarial vishing transcripts that evade detection while maintaining deceptive intent. We construct a systematic attack pipeline that employs prompt engineering and semantic obfuscation to transform real-world vishing scripts using four commercial LLMs. The generated transcripts are evaluated against multiple ML classifiers trained on a real-world Korean vishing dataset (KorCCViD) with statistical testing. Our experiments reveal that LLM-generated transcripts are both practically and statistically effective against ML-based classifiers. In particular, transcripts crafted by GPT-4o significantly reduce classifier accuracy (by up to 30.96%) while maintaining high semantic similarity, as measured by BERTScore. Moreover, these attacks are both time-efficient and cost-effective, with average generation times under 9 seconds and negligible financial cost per query. The results underscore the pressing need for more resilient vishing detection frameworks and highlight the imperative for LLM providers to enforce stronger safeguards against prompt misuse in adversarial social engineering contexts.



## **14. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2505.17579v2) [paper-pdf](http://arxiv.org/pdf/2505.17579v2)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Ishobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.



## **15. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

cs.CV

Accepted at SafeMM-AI @ ICCV 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2502.09110v2) [paper-pdf](http://arxiv.org/pdf/2502.09110v2)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.



## **16. Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models**

cs.CV

ACMMM 2025 Accepted

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16257v1) [paper-pdf](http://arxiv.org/pdf/2507.16257v1)

**Authors**: Futa Waseda, Saku Sugawara, Isao Echizen

**Abstract**: Defending pre-trained vision-language models (VLMs), such as CLIP, against adversarial attacks is crucial, as these models are widely used in diverse zero-shot tasks, including image classification. However, existing adversarial training (AT) methods for robust fine-tuning largely overlook the role of language in enhancing visual robustness. Specifically, (1) supervised AT methods rely on short texts (e.g., class labels) to generate adversarial perturbations, leading to overfitting to object classes in the training data, and (2) unsupervised AT avoids this overfitting but remains suboptimal against practical text-guided adversarial attacks due to its lack of semantic guidance. To address these limitations, we propose Quality Text-guided Adversarial Fine-Tuning (QT-AFT), which leverages high-quality captions during training to guide adversarial examples away from diverse semantics present in images. This enables the visual encoder to robustly recognize a broader range of image features even under adversarial noise, thereby enhancing robustness across diverse downstream tasks. QT-AFT overcomes the key weaknesses of prior methods -- overfitting in supervised AT and lack of semantic awareness in unsupervised AT -- achieving state-of-the-art zero-shot adversarial robustness and clean accuracy, evaluated across 16 zero-shot datasets. Furthermore, our comprehensive study uncovers several key insights into the role of language in enhancing vision robustness; for example, describing object properties in addition to object names further enhances zero-shot robustness. Our findings point to an urgent direction for future work -- centering high-quality linguistic supervision in robust visual representation learning.



## **17. Pulse-Level Simulation of Crosstalk Attacks on Superconducting Quantum Hardware**

quant-ph

This paper has been accepted to the Security, Privacy, and Resilience  Workshop at IEEE Quantum Week (QCE 2025) and will appear in the workshop  proceedings

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16181v1) [paper-pdf](http://arxiv.org/pdf/2507.16181v1)

**Authors**: Syed Emad Uddin Shubha, Tasnuva Farheen

**Abstract**: Hardware crosstalk in multi-tenant superconducting quantum computers poses a severe security threat, allowing adversaries to induce targeted errors across tenant boundaries by injecting carefully engineered pulses. We present a simulation-based study of active crosstalk attacks at the pulse level, analyzing how adversarial control of pulse timing, shape, amplitude, and coupling can disrupt a victim's computation. Our framework models the time-dependent dynamics of a three-qubit system in the rotating frame, capturing both always-on couplings and injected drive pulses. We examine two attack strategies: attacker-first (pulse before victim operation) and victim-first (pulse after), and systematically identify the pulse and coupling configurations that cause the largest logical errors. Protocol-level experiments on quantum coin flip and XOR classification circuits show that some protocols are highly vulnerable to these attacks, while others remain robust. Based on these findings, we discuss practical methods for detection and mitigation to improve security in quantum cloud platforms.



## **18. Attacking interpretable NLP systems**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16164v1) [paper-pdf](http://arxiv.org/pdf/2507.16164v1)

**Authors**: Eldor Abdukhamidov, Tamer Abuhmed, Joanna C. S. Santos, Mohammed Abuhamad

**Abstract**: Studies have shown that machine learning systems are vulnerable to adversarial examples in theory and practice. Where previous attacks have focused mainly on visual models that exploit the difference between human and machine perception, text-based models have also fallen victim to these attacks. However, these attacks often fail to maintain the semantic meaning of the text and similarity. This paper introduces AdvChar, a black-box attack on Interpretable Natural Language Processing Systems, designed to mislead the classifier while keeping the interpretation similar to benign inputs, thus exploiting trust in system transparency. AdvChar achieves this by making less noticeable modifications to text input, forcing the deep learning classifier to make incorrect predictions and preserve the original interpretation. We use an interpretation-focused scoring approach to determine the most critical tokens that, when changed, can cause the classifier to misclassify the input. We apply simple character-level modifications to measure the importance of tokens, minimizing the difference between the original and new text while generating adversarial interpretations similar to benign ones. We thoroughly evaluated AdvChar by testing it against seven NLP models and three interpretation models using benchmark datasets for the classification task. Our experiments show that AdvChar can significantly reduce the prediction accuracy of current deep learning models by altering just two characters on average in input samples.



## **19. DP2Guard: A Lightweight and Byzantine-Robust Privacy-Preserving Federated Learning Scheme for Industrial IoT**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16134v1) [paper-pdf](http://arxiv.org/pdf/2507.16134v1)

**Authors**: Baofu Han, Bing Li, Yining Qi, Raja Jurdak, Kaibin Huang, Chau Yuen

**Abstract**: Privacy-Preserving Federated Learning (PPFL) has emerged as a secure distributed Machine Learning (ML) paradigm that aggregates locally trained gradients without exposing raw data. To defend against model poisoning threats, several robustness-enhanced PPFL schemes have been proposed by integrating anomaly detection. Nevertheless, they still face two major challenges: (1) the reliance on heavyweight encryption techniques results in substantial communication and computation overhead; and (2) single-strategy defense mechanisms often fail to provide sufficient robustness against adaptive adversaries. To overcome these challenges, we propose DP2Guard, a lightweight PPFL framework that enhances both privacy and robustness. DP2Guard leverages a lightweight gradient masking mechanism to replace costly cryptographic operations while ensuring the privacy of local gradients. A hybrid defense strategy is proposed, which extracts gradient features using singular value decomposition and cosine similarity, and applies a clustering algorithm to effectively identify malicious gradients. Additionally, DP2Guard adopts a trust score-based adaptive aggregation scheme that adjusts client weights according to historical behavior, while blockchain records aggregated results and trust scores to ensure tamper-proof and auditable training. Extensive experiments conducted on two public datasets demonstrate that DP2Guard effectively defends against four advanced poisoning attacks while ensuring privacy with reduced communication and computation costs.



## **20. DP-TLDM: Differentially Private Tabular Latent Diffusion Model**

cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2403.07842v2) [paper-pdf](http://arxiv.org/pdf/2403.07842v2)

**Authors**: Chaoyi Zhu, Jiayi Tang, Juan F. Pérez, Marten van Dijk, Lydia Y. Chen

**Abstract**: Synthetic data from generative models emerges as the privacy-preserving data sharing solution. Such a synthetic data set shall resemble the original data without revealing identifiable private information. Till date, the prior focus on limited types of tabular synthesizers and a small number of privacy attacks, particularly on Generative Adversarial Networks, and overlooks membership inference attacks and defense strategies, i.e., differential privacy. Motivated by the conundrum of keeping high data quality and low privacy risk of synthetic data tables, we propose DPTLDM, Differentially Private Tabular Latent Diffusion Model, which is composed of an autoencoder network to encode the tabular data and a latent diffusion model to synthesize the latent tables. Following the emerging f-DP framework, we apply DP-SGD to train the auto-encoder in combination with batch clipping and use the separation value as the privacy metric to better capture the privacy gain from DP algorithms. Our empirical evaluation demonstrates that DPTLDM is capable of achieving a meaningful theoretical privacy guarantee while also significantly enhancing the utility of synthetic data. Specifically, compared to other DP-protected tabular generative models, DPTLDM improves the synthetic quality by an average of 35% in data resemblance, 15% in the utility for downstream tasks, and 50% in data discriminability, all while preserving a comparable level of privacy risk.



## **21. Erasing Conceptual Knowledge from Language Models**

cs.CL

Project Page: https://elm.baulab.info

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2410.02760v3) [paper-pdf](http://arxiv.org/pdf/2410.02760v3)

**Authors**: Rohit Gandikota, Sheridan Feucht, Samuel Marks, David Bau

**Abstract**: In this work, we introduce Erasure of Language Memory (ELM), a principled approach to concept-level unlearning that operates by matching distributions defined by the model's own introspective classification capabilities. Our key insight is that effective unlearning should leverage the model's ability to evaluate its own knowledge, using the language model itself as a classifier to identify and reduce the likelihood of generating content related to undesired concepts. ELM applies this framework to create targeted low-rank updates that reduce generation probabilities for concept-specific content while preserving the model's broader capabilities. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative evaluation reveals that ELM-modified models achieve near-random performance on assessments targeting erased concepts, while simultaneously preserving generation coherence, maintaining benchmark performance on unrelated tasks, and exhibiting strong robustness to adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info



## **22. Disrupting Semantic and Abstract Features for Better Adversarial Transferability**

cs.CV

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.16052v1) [paper-pdf](http://arxiv.org/pdf/2507.16052v1)

**Authors**: Yuyang Luo, Xiaosen Wang, Zhijin Ge, Yingzhe He

**Abstract**: Adversarial examples pose significant threats to deep neural networks (DNNs), and their property of transferability in the black-box setting has led to the emergence of transfer-based attacks, making it feasible to target real-world applications employing DNNs. Among them, feature-level attacks, where intermediate features are perturbed based on feature importance weight matrix computed from transformed images, have gained popularity. In this work, we find that existing feature-level attacks primarily manipulate the semantic information to derive the weight matrix. Inspired by several works that find CNNs tend to focus more on high-frequency components (a.k.a. abstract features, e.g., texture, edge, etc.), we validate that transforming images in the high-frequency space also improves transferability. Based on this finding, we propose a balanced approach called Semantic and Abstract FEatures disRuption (SAFER). Specifically, SAFER conducts BLOCKMIX on the input image and SELF-MIX on the frequency spectrum when computing the weight matrix to highlight crucial features. By using such a weight matrix, we can direct the attacker to disrupt both semantic and abstract features, leading to improved transferability. Extensive experiments on the ImageNet dataset also demonstrate the effectiveness of our method in boosting adversarial transferability.



## **23. Does More Inference-Time Compute Really Help Robustness?**

cs.AI

Preprint

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15974v1) [paper-pdf](http://arxiv.org/pdf/2507.15974v1)

**Authors**: Tong Wu, Chong Xiang, Jiachen T. Wang, Weichen Yu, Chawin Sitawarin, Vikash Sehwag, Prateek Mittal

**Abstract**: Recently, Zaremba et al. demonstrated that increasing inference-time computation improves robustness in large proprietary reasoning LLMs. In this paper, we first show that smaller-scale, open-source models (e.g., DeepSeek R1, Qwen3, Phi-reasoning) can also benefit from inference-time scaling using a simple budget forcing strategy. More importantly, we reveal and critically examine an implicit assumption in prior work: intermediate reasoning steps are hidden from adversaries. By relaxing this assumption, we identify an important security risk, intuitively motivated and empirically verified as an inverse scaling law: if intermediate reasoning steps become explicitly accessible, increased inference-time computation consistently reduces model robustness. Finally, we discuss practical scenarios where models with hidden reasoning chains are still vulnerable to attacks, such as models with tool-integrated reasoning and advanced reasoning extraction attacks. Our findings collectively demonstrate that the robustness benefits of inference-time scaling depend heavily on the adversarial setting and deployment context. We urge practitioners to carefully weigh these subtle trade-offs before applying inference-time scaling in security-sensitive, real-world applications.



## **24. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges**

cs.ET

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.06156v3) [paper-pdf](http://arxiv.org/pdf/2507.06156v3)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.



## **25. Sparsification Under Siege: Defending Against Poisoning Attacks in Communication-Efficient Federated Learning**

cs.CR

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2505.01454v4) [paper-pdf](http://arxiv.org/pdf/2505.01454v4)

**Authors**: Zhiyong Jin, Runhua Xu, Chao Li, Yizhong Liu, Jianxin Li, James Joshi

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy, yet it faces significant challenges in communication efficiency and vulnerability to poisoning attacks. While sparsification techniques mitigate communication overhead by transmitting only critical model parameters, they inadvertently amplify security risks: adversarial clients can exploit sparse updates to evade detection and degrade model performance. Existing defense mechanisms, designed for standard FL communication scenarios, are ineffective in addressing these vulnerabilities within sparsified FL. To bridge this gap, we propose FLARE, a novel federated learning framework that integrates sparse index mask inspection and model update sign similarity analysis to detect and mitigate poisoning attacks in sparsified FL. Extensive experiments across multiple datasets and adversarial scenarios demonstrate that FLARE significantly outperforms existing defense strategies, effectively securing sparsified FL against poisoning attacks while maintaining communication efficiency.



## **26. Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems**

cs.CR

26 pages

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15613v1) [paper-pdf](http://arxiv.org/pdf/2507.15613v1)

**Authors**: Andrii Balashov, Olena Ponomarova, Xiaohua Zhai

**Abstract**: Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place.   We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.



## **27. Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI**

cs.CV

CVPR 2025 (poster), 19 pages, 5 figures

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2411.15265v2) [paper-pdf](http://arxiv.org/pdf/2411.15265v2)

**Authors**: Won Jun Kim, Hyungjin Chung, Jaemin Kim, Sangmin Lee, Byeongsu Sim, Jong Chul Ye

**Abstract**: Gradient-based methods are a prototypical family of explainability techniques, especially for image-based models. Nonetheless, they have several shortcomings in that they (1) require white-box access to models, (2) are vulnerable to adversarial attacks, and (3) produce attributions that lie off the image manifold, leading to explanations that are not actually faithful to the model and do not align well with human perception. To overcome these challenges, we introduce Derivative-Free Diffusion Manifold-Constrainted Gradients (FreeMCG), a novel method that serves as an improved basis for explainability of a given neural network than the traditional gradient. Specifically, by leveraging ensemble Kalman filters and diffusion models, we derive a derivative-free approximation of the model's gradient projected onto the data manifold, requiring access only to the model's outputs. We demonstrate the effectiveness of FreeMCG by applying it to both counterfactual generation and feature attribution, which have traditionally been treated as distinct tasks. Through comprehensive evaluation on both tasks, counterfactual explanation and feature attribution, we show that our method yields state-of-the-art results while preserving the essential properties expected of XAI tools.



## **28. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

cs.CV

This paper is accepted by ACM MM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2405.20090v5) [paper-pdf](http://arxiv.org/pdf/2405.20090v5)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.



## **29. Scaling Decentralized Learning with FLock**

cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15349v1) [paper-pdf](http://arxiv.org/pdf/2507.15349v1)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.



## **30. Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models**

cs.CL

Accepted to CoLM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2503.01781v2) [paper-pdf](http://arxiv.org/pdf/2503.01781v2)

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.



## **31. Defective Convolutional Networks**

cs.CV

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/1911.08432v3) [paper-pdf](http://arxiv.org/pdf/1911.08432v3)

**Authors**: Tiange Luo, Tianle Cai, Mengxiao Zhang, Siyu Chen, Di He, Liwei Wang

**Abstract**: Robustness of convolutional neural networks (CNNs) has gained in importance on account of adversarial examples, i.e., inputs added as well-designed perturbations that are imperceptible to humans but can cause the model to predict incorrectly. Recent research suggests that the noises in adversarial examples break the textural structure, which eventually leads to wrong predictions. To mitigate the threat of such adversarial attacks, we propose defective convolutional networks that make predictions relying less on textural information but more on shape information by properly integrating defective convolutional layers into standard CNNs. The defective convolutional layers contain defective neurons whose activations are set to be a constant function. As defective neurons contain no information and are far different from standard neurons in its spatial neighborhood, the textural features cannot be accurately extracted, and so the model has to seek other features for classification, such as the shape. We show extensive evidence to justify our proposal and demonstrate that defective CNNs can defense against black-box attacks better than standard CNNs. In particular, they achieve state-of-the-art performance against transfer-based attacks without any adversarial training being applied.



## **32. ROBAD: Robust Adversary-aware Local-Global Attended Bad Actor Detection Sequential Model**

cs.LG

15 pages, 12 tables

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.15067v1) [paper-pdf](http://arxiv.org/pdf/2507.15067v1)

**Authors**: Bing He, Mustaque Ahamad, Srijan Kumar

**Abstract**: Detecting bad actors is critical to ensure the safety and integrity of internet platforms. Several deep learning-based models have been developed to identify such users. These models should not only accurately detect bad actors, but also be robust against adversarial attacks that aim to evade detection. However, past deep learning-based detection models do not meet the robustness requirement because they are sensitive to even minor changes in the input sequence. To address this issue, we focus on (1) improving the model understanding capability and (2) enhancing the model knowledge such that the model can recognize potential input modifications when making predictions. To achieve these goals, we create a novel transformer-based classification model, called ROBAD (RObust adversary-aware local-global attended Bad Actor Detection model), which uses the sequence of user posts to generate user embedding to detect bad actors. Particularly, ROBAD first leverages the transformer encoder block to encode each post bidirectionally, thus building a post embedding to capture the local information at the post level. Next, it adopts the transformer decoder block to model the sequential pattern in the post embeddings by using the attention mechanism, which generates the sequence embedding to obtain the global information at the sequence level. Finally, to enrich the knowledge of the model, embeddings of modified sequences by mimicked attackers are fed into a contrastive-learning-enhanced classification layer for sequence prediction. In essence, by capturing the local and global information (i.e., the post and sequence information) and leveraging the mimicked behaviors of bad actors in training, ROBAD can be robust to adversarial attacks. Extensive experiments on Yelp and Wikipedia datasets show that ROBAD can effectively detect bad actors when under state-of-the-art adversarial attacks.



## **33. DeRAG: Black-box Adversarial Attacks on Multiple Retrieval-Augmented Generation Applications via Prompt Injection**

cs.AI

Accepted by KDD Workshop on Prompt Optimization 2025

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.15042v1) [paper-pdf](http://arxiv.org/pdf/2507.15042v1)

**Authors**: Jerry Wang, Fang Yu

**Abstract**: Adversarial prompt attacks can significantly alter the reliability of Retrieval-Augmented Generation (RAG) systems by re-ranking them to produce incorrect outputs. In this paper, we present a novel method that applies Differential Evolution (DE) to optimize adversarial prompt suffixes for RAG-based question answering. Our approach is gradient-free, treating the RAG pipeline as a black box and evolving a population of candidate suffixes to maximize the retrieval rank of a targeted incorrect document to be closer to real world scenarios. We conducted experiments on the BEIR QA datasets to evaluate attack success at certain retrieval rank thresholds under multiple retrieving applications. Our results demonstrate that DE-based prompt optimization attains competitive (and in some cases higher) success rates compared to GGPP to dense retrievers and PRADA to sparse retrievers, while using only a small number of tokens (<=5 tokens) in the adversarial suffix. Furthermore, we introduce a readability-aware suffix construction strategy, validated by a statistically significant reduction in MLM negative log-likelihood with Welch's t-test. Through evaluations with a BERT-based adversarial suffix detector, we show that DE-generated suffixes evade detection, yielding near-chance detection accuracy.



## **34. Adversarial Destabilization Attacks to Direct Data-Driven Control**

eess.SY

15 pages

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14863v1) [paper-pdf](http://arxiv.org/pdf/2507.14863v1)

**Authors**: Hampei Sasahara

**Abstract**: This study investigates the vulnerability of direct data-driven control methods, specifically for the linear quadratic regulator problem, to adversarial perturbations in collected data used for controller synthesis. We consider stealthy attacks that subtly manipulate offline-collected data to destabilize the resulting closed-loop system while evading detection. To generate such perturbations, we propose the Directed Gradient Sign Method (DGSM) and its iterative variant (I-DGSM), adaptations of the fast gradient sign method originally developed for neural networks, which align perturbations with the gradient of the spectral radius of the closed-loop matrix to reduce stability. A key contribution is an efficient gradient computation technique based on implicit differentiation through the Karush-Kuhn-Tucker conditions of the underlying semidefinite program, enabling scalable and exact gradient evaluation without repeated optimization computations. To defend against these attacks, we propose two defense strategies: a regularization-based approach that enhances robustness by suppressing controller sensitivity to data perturbations and a robust data-driven control approach that guarantees closed-loop stability within bounded perturbation sets. Extensive numerical experiments on benchmark systems show that adversarial perturbations with magnitudes up to ten times smaller than random noise can destabilize controllers trained on corrupted data and that the proposed defense strategies effectively mitigate attack success rates while maintaining control performance. Additionally, we evaluate attack transferability under partial knowledge scenarios, highlighting the practical importance of protecting training data confidentiality.



## **35. Data-Plane Telemetry to Mitigate Long-Distance BGP Hijacks**

cs.NI

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14842v1) [paper-pdf](http://arxiv.org/pdf/2507.14842v1)

**Authors**: Satadal Sengupta, Hyojoon Kim, Daniel Jubas, Maria Apostolaki, Jennifer Rexford

**Abstract**: Poor security of Internet routing enables adversaries to divert user data through unintended infrastructures (hijack). Of particular concern -- and the focus of this paper -- are cases where attackers reroute domestic traffic through foreign countries, exposing it to surveillance, bypassing legal privacy protections, and posing national security threats. Efforts to detect and mitigate such attacks have focused primarily on the control plane while data-plane signals remain largely overlooked. In particular, change in propagation delay caused by rerouting offers a promising signal: the change is unavoidable and the increased propagation delay is directly observable from the affected networks. In this paper, we explore the practicality of using delay variations for hijack detection, addressing two key questions: (1) What coverage can this provide, given its heavy dependence on the geolocations of the sender, receiver, and adversary? and (2) Can an always-on latency-based detection system be deployed without disrupting normal network operations? We observe that for 86% of victim-attacker country pairs in the world, mid-attack delays exceed pre-attack delays by at least 25% in real deployments, making delay-based hijack detection promising. To demonstrate practicality, we design HiDe, which reliably detects delay surges from long-distance hijacks at line rate. We measure HiDe's accuracy and false-positive rate on real-world data and validate it with ethically conducted hijacks.



## **36. Manipulating LLM Web Agents with Indirect Prompt Injection Attack via HTML Accessibility Tree**

cs.CR

EMNLP 2025 System Demonstrations Submission

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14799v1) [paper-pdf](http://arxiv.org/pdf/2507.14799v1)

**Authors**: Sam Johnson, Viet Pham, Thai Le

**Abstract**: This work demonstrates that LLM-based web navigation agents offer powerful automation capabilities but are vulnerable to Indirect Prompt Injection (IPI) attacks. We show that adversaries can embed universal adversarial triggers in webpage HTML to hijack agent behavior that utilizes the accessibility tree to parse HTML, causing unintended or malicious actions. Using the Greedy Coordinate Gradient (GCG) algorithm and a Browser Gym agent powered by Llama-3.1, our system demonstrates high success rates across real websites in both targeted and general attacks, including login credential exfiltration and forced ad clicks. Our empirical results highlight critical security risks and the need for stronger defenses as LLM-driven autonomous web agents become more widely adopted. The system software (https://github.com/sej2020/manipulating-web-agents) is released under the MIT License, with an accompanying publicly available demo website (http://lethaiq.github.io/attack-web-llm-agent).



## **37. GCC-Spam: Spam Detection via GAN, Contrastive Learning, and Character Similarity Networks**

cs.LG

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2507.14679v1) [paper-pdf](http://arxiv.org/pdf/2507.14679v1)

**Authors**: Zixin Xu, Zhijie Wang, Zhiyuan Pan

**Abstract**: The exponential growth of spam text on the Internet necessitates robust detection mechanisms to mitigate risks such as information leakage and social instability. This work addresses two principal challenges: adversarial strategies employed by spammers and the scarcity of labeled data. We propose a novel spam-text detection framework GCC-Spam, which integrates three core innovations. First, a character similarity network captures orthographic and phonetic features to counter character-obfuscation attacks and furthermore produces sentence embeddings for downstream classification. Second, contrastive learning enhances discriminability by optimizing the latent-space distance between spam and normal texts. Third, a Generative Adversarial Network (GAN) generates realistic pseudo-spam samples to alleviate data scarcity while improving model robustness and classification accuracy. Extensive experiments on real-world datasets demonstrate that our model outperforms baseline approaches, achieving higher detection rates with significantly fewer labeled examples.



## **38. VTarbel: Targeted Label Attack with Minimal Knowledge on Detector-enhanced Vertical Federated Learning**

cs.CR

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2507.14625v1) [paper-pdf](http://arxiv.org/pdf/2507.14625v1)

**Authors**: Juntao Tan, Anran Li, Quanchao Liu, Peng Ran, Lan Zhang

**Abstract**: Vertical federated learning (VFL) enables multiple parties with disjoint features to collaboratively train models without sharing raw data. While privacy vulnerabilities of VFL are extensively-studied, its security threats-particularly targeted label attacks-remain underexplored. In such attacks, a passive party perturbs inputs at inference to force misclassification into adversary-chosen labels. Existing methods rely on unrealistic assumptions (e.g., accessing VFL-model's outputs) and ignore anomaly detectors deployed in real-world systems. To bridge this gap, we introduce VTarbel, a two-stage, minimal-knowledge attack framework explicitly designed to evade detector-enhanced VFL inference. During the preparation stage, the attacker selects a minimal set of high-expressiveness samples (via maximum mean discrepancy), submits them through VFL protocol to collect predicted labels, and uses these pseudo-labels to train estimated detector and surrogate model on local features. In attack stage, these models guide gradient-based perturbations of remaining samples, crafting adversarial instances that induce targeted misclassifications and evade detection. We implement VTarbel and evaluate it against four model architectures, seven multimodal datasets, and two anomaly detectors. Across all settings, VTarbel outperforms four state-of-the-art baselines, evades detection, and retains effective against three representative privacy-preserving defenses. These results reveal critical security blind spots in current VFL deployments and underscore urgent need for robust, attack-aware defenses.



## **39. 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving**

cs.CV

Submitted to WACV 2026

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2507.09993v2) [paper-pdf](http://arxiv.org/pdf/2507.09993v2)

**Authors**: Yixun Zhang, Lizhi Wang, Junjun Zhao, Wending Zhao, Feng Zhou, Yonghao Dang, Jianqin Yin

**Abstract**: Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. Existing 2D and 3D physical attacks, due to their focus on texture optimization, often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture optimization, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module that filters outliers to preserve geometric fidelity, and a physical augmentation module that simulates complex physical scenarios to enhance attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21\% to 7.38\%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks.



## **40. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

See also followup work at arXiv:2407.15549

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2403.05030v5) [paper-pdf](http://arxiv.org/pdf/2403.05030v5)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without leveraging knowledge of what they are or using inputs that elicit them. LAT makes use of the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. Here, we use it to defend against failure modes without examples that elicit them. Specifically, we use LAT to remove backdoors and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **41. FedStrategist: A Meta-Learning Framework for Adaptive and Robust Aggregation in Federated Learning**

cs.LG

24 pages, 8 figures. This work is intended for a journal submission

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.14322v1) [paper-pdf](http://arxiv.org/pdf/2507.14322v1)

**Authors**: Md Rafid Haque, Abu Raihan Mostofa Kamal, Md. Azam Hossain

**Abstract**: Federated Learning (FL) offers a paradigm for privacy-preserving collaborative AI, but its decentralized nature creates significant vulnerabilities to model poisoning attacks. While numerous static defenses exist, their effectiveness is highly context-dependent, often failing against adaptive adversaries or in heterogeneous data environments. This paper introduces FedStrategist, a novel meta-learning framework that reframes robust aggregation as a real-time, cost-aware control problem. We design a lightweight contextual bandit agent that dynamically selects the optimal aggregation rule from an arsenal of defenses based on real-time diagnostic metrics. Through comprehensive experiments, we demonstrate that no single static rule is universally optimal. We show that our adaptive agent successfully learns superior policies across diverse scenarios, including a ``Krum-favorable" environment and against a sophisticated "stealth" adversary designed to neutralize specific diagnostic signals. Critically, we analyze the paradoxical scenario where a non-robust baseline achieves high but compromised accuracy, and demonstrate that our agent learns a conservative policy to prioritize model integrity. Furthermore, we prove the agent's policy is controllable via a single "risk tolerance" parameter, allowing practitioners to explicitly manage the trade-off between performance and security. Our work provides a new, practical, and analyzable approach to creating resilient and intelligent decentralized AI systems.



## **42. An Adversarial-Driven Experimental Study on Deep Learning for RF Fingerprinting**

cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.14109v1) [paper-pdf](http://arxiv.org/pdf/2507.14109v1)

**Authors**: Xinyu Cao, Bimal Adhikari, Shangqing Zhao, Jingxian Wu, Yanjun Pan

**Abstract**: Radio frequency (RF) fingerprinting, which extracts unique hardware imperfections of radio devices, has emerged as a promising physical-layer device identification mechanism in zero trust architectures and beyond 5G networks. In particular, deep learning (DL) methods have demonstrated state-of-the-art performance in this domain. However, existing approaches have primarily focused on enhancing system robustness against temporal and spatial variations in wireless environments, while the security vulnerabilities of these DL-based approaches have often been overlooked. In this work, we systematically investigate the security risks of DL-based RF fingerprinting systems through an adversarial-driven experimental analysis. We observe a consistent misclassification behavior for DL models under domain shifts, where a device is frequently misclassified as another specific one. Our analysis based on extensive real-world experiments demonstrates that this behavior can be exploited as an effective backdoor to enable external attackers to intrude into the system. Furthermore, we show that training DL models on raw received signals causes the models to entangle RF fingerprints with environmental and signal-pattern features, creating additional attack vectors that cannot be mitigated solely through post-processing security methods such as confidence thresholds.



## **43. CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors**

cs.CV

Accepted by ACMMM 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2504.10888v2) [paper-pdf](http://arxiv.org/pdf/2504.10888v2)

**Authors**: Jiahuan Long, Wen Yao, Tingsong Jiang, Chao Ma

**Abstract**: Adversarial patches are widely used to evaluate the robustness of object detection systems in real-world scenarios. These patches were initially designed to deceive single-modal detectors (e.g., visible or infrared) and have recently been extended to target visible-infrared dual-modal detectors. However, existing dual-modal adversarial patch attacks have limited attack effectiveness across diverse physical scenarios. To address this, we propose CDUPatch, a universal cross-modal patch attack against visible-infrared object detectors across scales, views, and scenarios. Specifically, we observe that color variations lead to different levels of thermal absorption, resulting in temperature differences in infrared imaging. Leveraging this property, we propose an RGB-to-infrared adapter that maps RGB patches to infrared patches, enabling unified optimization of cross-modal patches. By learning an optimal color distribution on the adversarial patch, we can manipulate its thermal response and generate an adversarial infrared texture. Additionally, we introduce a multi-scale clipping strategy and construct a new visible-infrared dataset, MSDrone, which contains aerial vehicle images in varying scales and perspectives. These data augmentation strategies enhance the robustness of our patch in real-world conditions. Experiments on four benchmark datasets (e.g., DroneVehicle, LLVIP, VisDrone, MSDrone) show that our method outperforms existing patch attacks in the digital domain. Extensive physical tests further confirm strong transferability across scales, views, and scenarios.



## **44. Bridging Local and Global Knowledge via Transformer in Board Games**

cs.LG

Accepted by the Thirty-Fourth International Joint Conferences on  Artificial Intelligence (IJCAI-25)

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2410.05347v2) [paper-pdf](http://arxiv.org/pdf/2410.05347v2)

**Authors**: Yan-Ru Ju, Tai-Lin Wu, Chung-Chin Shih, Ti-Rong Wu

**Abstract**: Although AlphaZero has achieved superhuman performance in board games, recent studies reveal its limitations in handling scenarios requiring a comprehensive understanding of the entire board, such as recognizing long-sequence patterns in Go. To address this challenge, we propose ResTNet, a network that interleaves residual and Transformer blocks to bridge local and global knowledge. ResTNet improves playing strength across multiple board games, increasing win rate from 54.6% to 60.8% in 9x9 Go, 53.6% to 60.9% in 19x19 Go, and 50.4% to 58.0% in 19x19 Hex. In addition, ResTNet effectively processes global information and tackles two long-sequence patterns in 19x19 Go, including circular pattern and ladder pattern. It reduces the mean square error for circular pattern recognition from 2.58 to 1.07 and lowers the attack probability against an adversary program from 70.44% to 23.91%. ResTNet also improves ladder pattern recognition accuracy from 59.15% to 80.01%. By visualizing attention maps, we demonstrate that ResTNet captures critical game concepts in both Go and Hex, offering insights into AlphaZero's decision-making process. Overall, ResTNet shows a promising approach to integrating local and global knowledge, paving the way for more effective AlphaZero-based algorithms in board games. Our code is available at https://rlg.iis.sinica.edu.tw/papers/restnet.



## **45. From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios**

cs.AI

Final Version and Paper Accepted at IEEE ITSC 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2502.02145v4) [paper-pdf](http://arxiv.org/pdf/2502.02145v4)

**Authors**: Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, Johannes Betz

**Abstract**: Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.



## **46. Adversarial Training Improves Generalization Under Distribution Shifts in Bioacoustics**

cs.LG

Work in progress

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.13727v1) [paper-pdf](http://arxiv.org/pdf/2507.13727v1)

**Authors**: René Heinrich, Lukas Rauch, Bernhard Sick, Christoph Scholz

**Abstract**: Adversarial training is a promising strategy for enhancing model robustness against adversarial attacks. However, its impact on generalization under substantial data distribution shifts in audio classification remains largely unexplored. To address this gap, this work investigates how different adversarial training strategies improve generalization performance and adversarial robustness in audio classification. The study focuses on two model architectures: a conventional convolutional neural network (ConvNeXt) and an inherently interpretable prototype-based model (AudioProtoPNet). The approach is evaluated using a challenging bird sound classification benchmark. This benchmark is characterized by pronounced distribution shifts between training and test data due to varying environmental conditions and recording methods, a common real-world challenge. The investigation explores two adversarial training strategies: one based on output-space attacks that maximize the classification loss function, and another based on embedding-space attacks designed to maximize embedding dissimilarity. These attack types are also used for robustness evaluation. Additionally, for AudioProtoPNet, the study assesses the stability of its learned prototypes under targeted embedding-space attacks. Results show that adversarial training, particularly using output-space attacks, improves clean test data performance by an average of 10.5% relative and simultaneously strengthens the adversarial robustness of the models. These findings, although derived from the bird sound domain, suggest that adversarial training holds potential to enhance robustness against both strong distribution shifts and adversarial attacks in challenging audio classification settings.



## **47. Breaking the Illusion of Security via Interpretation: Interpretable Vision Transformer Systems under Attack**

cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.14248v1) [paper-pdf](http://arxiv.org/pdf/2507.14248v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Hyoungshick Kim, Tamer Abuhmed

**Abstract**: Vision transformer (ViT) models, when coupled with interpretation models, are regarded as secure and challenging to deceive, making them well-suited for security-critical domains such as medical applications, autonomous vehicles, drones, and robotics. However, successful attacks on these systems can lead to severe consequences. Recent research on threats targeting ViT models primarily focuses on generating the smallest adversarial perturbations that can deceive the models with high confidence, without considering their impact on model interpretations. Nevertheless, the use of interpretation models can effectively assist in detecting adversarial examples. This study investigates the vulnerability of transformer models to adversarial attacks, even when combined with interpretation models. We propose an attack called "AdViT" that generates adversarial examples capable of misleading both a given transformer model and its coupled interpretation model. Through extensive experiments on various transformer models and two transformer-based interpreters, we demonstrate that AdViT achieves a 100% attack success rate in both white-box and black-box scenarios. In white-box scenarios, it reaches up to 98% misclassification confidence, while in black-box scenarios, it reaches up to 76% misclassification confidence. Remarkably, AdViT consistently generates accurate interpretations in both scenarios, making the adversarial examples more difficult to detect.



## **48. GIFT: Gradient-aware Immunization of diffusion models against malicious Fine-Tuning with safe concepts retention**

cs.CR

Warning: This paper contains NSFW content. Reader discretion is  advised

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.13598v1) [paper-pdf](http://arxiv.org/pdf/2507.13598v1)

**Authors**: Amro Abdalla, Ismail Shaheen, Dan DeGenaro, Rupayan Mallick, Bogdan Raita, Sarah Adel Bargal

**Abstract**: We present GIFT: a {G}radient-aware {I}mmunization technique to defend diffusion models against malicious {F}ine-{T}uning while preserving their ability to generate safe content. Existing safety mechanisms like safety checkers are easily bypassed, and concept erasure methods fail under adversarial fine-tuning. GIFT addresses this by framing immunization as a bi-level optimization problem: the upper-level objective degrades the model's ability to represent harmful concepts using representation noising and maximization, while the lower-level objective preserves performance on safe data. GIFT achieves robust resistance to malicious fine-tuning while maintaining safe generative quality. Experimental results show that our method significantly impairs the model's ability to re-learn harmful concepts while maintaining performance on safe content, offering a promising direction for creating inherently safer generative models resistant to adversarial fine-tuning attacks.



## **49. BLAST: A Stealthy Backdoor Leverage Attack against Cooperative Multi-Agent Deep Reinforcement Learning based Systems**

cs.AI

12. arXiv admin note: substantial text overlap with arXiv:2409.07775

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2501.01593v2) [paper-pdf](http://arxiv.org/pdf/2501.01593v2)

**Authors**: Jing Fang, Saihao Yan, Xueyu Yin, Yinbo Yu, Chunwei Tian, Jiajia Liu

**Abstract**: Recent studies have shown that cooperative multi-agent deep reinforcement learning (c-MADRL) is under the threat of backdoor attacks. Once a backdoor trigger is observed, it will perform malicious actions leading to failures or malicious goals. However, existing backdoor attacks suffer from several issues, e.g., instant trigger patterns lack stealthiness, the backdoor is trained or activated by an additional network, or all agents are backdoored. To this end, in this paper, we propose a novel backdoor leverage attack against c-MADRL, BLAST, which attacks the entire multi-agent team by embedding the backdoor only in a single agent. Firstly, we introduce adversary spatiotemporal behavior patterns as the backdoor trigger rather than manual-injected fixed visual patterns or instant status and control the period to perform malicious actions. This method can guarantee the stealthiness and practicality of BLAST. Secondly, we hack the original reward function of the backdoor agent via unilateral guidance to inject BLAST, so as to achieve the \textit{leverage attack effect} that can pry open the entire multi-agent system via a single backdoor agent. We evaluate our BLAST against 3 classic c-MADRL algorithms (VDN, QMIX, and MAPPO) in 2 popular c-MADRL environments (SMAC and Pursuit), and 2 existing defense mechanisms. The experimental results demonstrate that BLAST can achieve a high attack success rate while maintaining a low clean performance variance rate.



## **50. STACK: Adversarial Attacks on LLM Safeguard Pipelines**

cs.CL

Fixed typos (including Figure 1), amended GPU-hours rather than days,  clarified ReNeLLM prompt modifications

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2506.24068v2) [paper-pdf](http://arxiv.org/pdf/2506.24068v2)

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.



