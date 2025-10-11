# Latest Adversarial Attack Papers
**update at 2025-10-11 09:51:27**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. AutoRed: A Free-form Adversarial Prompt Generation Framework for Automated Red Teaming**

cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08329v1) [paper-pdf](http://arxiv.org/pdf/2510.08329v1)

**Authors**: Muxi Diao, Yutao Mou, Keqing He, Hanbo Song, Lulu Zhao, Shikun Zhang, Wei Ye, Kongming Liang, Zhanyu Ma

**Abstract**: The safety of Large Language Models (LLMs) is crucial for the development of trustworthy AI applications. Existing red teaming methods often rely on seed instructions, which limits the semantic diversity of the synthesized adversarial prompts. We propose AutoRed, a free-form adversarial prompt generation framework that removes the need for seed instructions. AutoRed operates in two stages: (1) persona-guided adversarial instruction generation, and (2) a reflection loop to iteratively refine low-quality prompts. To improve efficiency, we introduce a verifier to assess prompt harmfulness without querying the target models. Using AutoRed, we build two red teaming datasets -- AutoRed-Medium and AutoRed-Hard -- and evaluate eight state-of-the-art LLMs. AutoRed achieves higher attack success rates and better generalization than existing baselines. Our results highlight the limitations of seed-based approaches and demonstrate the potential of free-form red teaming for LLM safety evaluation. We will open source our datasets in the near future.



## **2. Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning**

cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2505.16567v3) [paper-pdf](http://arxiv.org/pdf/2505.16567v3)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning open-weight Large Language Models (LLMs) is standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets leads to predictable behaviors. In this paper, we demonstrate, for the first time, that an adversary can create compromised LLMs that are performant and benign, yet exhibit adversarial behaviors once finetuned by downstream users. To this end, we propose an attack, FAB (Finetuning-activated Adversarial Behaviors), which compromises an LLM via meta-learning techniques that simulate downstream finetuning, explicitly optimizing for the emergence of adversarial behaviors in the finetuned models. At the same time, the compromised LLM is regularized to retain general capabilities and to exhibit no adversarial behaviors prior to finetuning. As a result, when users finetune (e.g., instruction-tuning, distillation, DPO) the seemingly benign model on their own datasets, they unknowingly trigger its dormant adversarial behavior. We experimentally demonstrate the effectiveness of FAB across multiple LLMs and three commonly considered target behaviors: unsolicited advertising, jailbreakability, and over-refusal. We show that FAB-triggers are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler, post-training algorithm). Our findings challenge prevailing assumptions on the security of finetuning, revealing a critical attack vector.



## **3. The Alignment Waltz: Jointly Training Agents to Collaborate for Safety**

cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08240v1) [paper-pdf](http://arxiv.org/pdf/2510.08240v1)

**Authors**: Jingyu Zhang, Haozhu Wang, Eric Michael Smith, Sid Wang, Amr Sharaf, Mahesh Pasupuleti, Benjamin Van Durme, Daniel Khashabi, Jason Weston, Hongyuan Zhan

**Abstract**: Harnessing the power of LLMs requires a delicate dance between being helpful and harmless. This creates a fundamental tension between two competing challenges: vulnerability to adversarial attacks that elicit unsafe content, and a tendency for overrefusal on benign but sensitive prompts. Current approaches often navigate this dance with safeguard models that completely reject any content that contains unsafe portions. This approach cuts the music entirely-it may exacerbate overrefusals and fails to provide nuanced guidance for queries it refuses. To teach models a more coordinated choreography, we propose WaltzRL, a novel multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. WaltzRL jointly trains a conversation agent and a feedback agent, where the latter is incentivized to provide useful suggestions that improve the safety and helpfulness of the conversation agent's responses. At the core of WaltzRL is a Dynamic Improvement Reward (DIR) that evolves over time based on how well the conversation agent incorporates the feedback. At inference time, unsafe or overrefusing responses from the conversation agent are improved rather than discarded. The feedback agent is deployed together with the conversation agent and only engages adaptively when needed, preserving helpfulness and low latency on safe queries. Our experiments, conducted across five diverse datasets, demonstrate that WaltzRL significantly reduces both unsafe responses (e.g., from 39.0% to 4.6% on WildJailbreak) and overrefusals (from 45.3% to 9.9% on OR-Bench) compared to various baselines. By enabling the conversation and feedback agents to co-evolve and adaptively apply feedback, WaltzRL enhances LLM safety without degrading general capabilities, thereby advancing the Pareto front between helpfulness and harmlessness.



## **4. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2507.11112v2) [paper-pdf](http://arxiv.org/pdf/2507.11112v2)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.



## **5. Interpreting LLM-as-a-Judge Policies via Verifiable Global Explanations**

cs.CL

12 pages, 2 figures, 3 tables

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08120v1) [paper-pdf](http://arxiv.org/pdf/2510.08120v1)

**Authors**: Jasmina Gajcin, Erik Miehling, Rahul Nair, Elizabeth Daly, Radu Marinescu, Seshu Tirupathi

**Abstract**: Using LLMs to evaluate text, that is, LLM-as-a-judge, is increasingly being used at scale to augment or even replace human annotations. As such, it is imperative that we understand the potential biases and risks of doing so. In this work, we propose an approach for extracting high-level concept-based global policies from LLM-as-a-Judge. Our approach consists of two algorithms: 1) CLoVE (Contrastive Local Verifiable Explanations), which generates verifiable, concept-based, contrastive local explanations and 2) GloVE (Global Verifiable Explanations), which uses iterative clustering, summarization and verification to condense local rules into a global policy. We evaluate GloVE on seven standard benchmarking datasets for content harm detection. We find that the extracted global policies are highly faithful to decisions of the LLM-as-a-Judge. Additionally, we evaluated the robustness of global policies to text perturbations and adversarial attacks. Finally, we conducted a user study to evaluate user understanding and satisfaction with global policies.



## **6. Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks**

cs.CL

Minor correction: Fixed sign errors in the results table. The update  does not affect the main findings or conclusions

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2506.11113v3) [paper-pdf](http://arxiv.org/pdf/2506.11113v3)

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.



## **7. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm**

cs.CL

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.15550v2) [paper-pdf](http://arxiv.org/pdf/2509.15550v2)

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao

**Abstract**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets. Code and data are available at https://github.com/Xiaoweizhu57/DNA-DetectLLM.



## **8. Backdoor Vectors: a Task Arithmetic View on Backdoor Attacks and Defenses**

cs.LG

22 pages, 13 figures, 15 tables

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08016v1) [paper-pdf](http://arxiv.org/pdf/2510.08016v1)

**Authors**: Stanisław Pawlak, Jan Dubiński, Daniel Marczak, Bartłomiej Twardowski

**Abstract**: Model merging (MM) recently emerged as an effective method for combining large deep learning models. However, it poses significant security risks. Recent research shows that it is highly susceptible to backdoor attacks, which introduce a hidden trigger into a single fine-tuned model instance that allows the adversary to control the output of the final merged model at inference time. In this work, we propose a simple framework for understanding backdoor attacks by treating the attack itself as a task vector. $Backdoor\ Vector\ (BV)$ is calculated as the difference between the weights of a fine-tuned backdoored model and fine-tuned clean model. BVs reveal new insights into attacks understanding and a more effective framework to measure their similarity and transferability. Furthermore, we propose a novel method that enhances backdoor resilience through merging dubbed $Sparse\ Backdoor\ Vector\ (SBV)$ that combines multiple attacks into a single one. We identify the core vulnerability behind backdoor threats in MM: $inherent\ triggers$ that exploit adversarial weaknesses in the base model. To counter this, we propose $Injection\ BV\ Subtraction\ (IBVS)$ - an assumption-free defense against backdoors in MM. Our results show that SBVs surpass prior attacks and is the first method to leverage merging to improve backdoor effectiveness. At the same time, IBVS provides a lightweight, general defense that remains effective even when the backdoor threat is entirely unknown.



## **9. Fewer Weights, More Problems: A Practical Attack on LLM Pruning**

cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07985v1) [paper-pdf](http://arxiv.org/pdf/2510.07985v1)

**Authors**: Kazuki Egashira, Robin Staab, Thibaud Gloaguen, Mark Vero, Martin Vechev

**Abstract**: Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.



## **10. Safe-Control: A Safety Patch for Mitigating Unsafe Content in Text-to-Image Generation Models**

cs.CV

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2508.21099v2) [paper-pdf](http://arxiv.org/pdf/2508.21099v2)

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo

**Abstract**: Despite the advancements in Text-to-Image (T2I) generation models, their potential for misuse or even abuse raises serious safety concerns. Model developers have made tremendous efforts to introduce safety mechanisms that can address these concerns in T2I models. However, the existing safety mechanisms, whether external or internal, either remain susceptible to evasion under distribution shifts or require extensive model-specific adjustments. To address these limitations, we introduce Safe-Control, an innovative plug-and-play safety patch designed to mitigate unsafe content generation in T2I models. Using data-driven strategies and safety-aware conditions, Safe-Control injects safety control signals into the locked T2I model, acting as an update in a patch-like manner. Model developers can also construct various safety patches to meet the evolving safety requirements, which can be flexibly merged into a single, unified patch. Its plug-and-play design further ensures adaptability, making it compatible with other T2I models of similar denoising architecture. We conduct extensive evaluations on six diverse and public T2I models. Empirical results highlight that Safe-Control is effective in reducing unsafe content generation across six diverse T2I models with similar generative architectures, yet it successfully maintains the quality and text alignment of benign images. Compared to seven state-of-the-art safety mechanisms, including both external and internal defenses, Safe-Control significantly outperforms all baselines in reducing unsafe content generation. For example, it reduces the probability of unsafe content generation to 7%, compared to approximately 20% for most baseline methods, under both unsafe prompts and the latest adversarial attacks.



## **11. Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor**

eess.AS

5 pages, 3 figures

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07909v1) [paper-pdf](http://arxiv.org/pdf/2510.07909v1)

**Authors**: Kuan-Yu Chen, Yi-Cheng Lin, Jeng-Lin Li, Jian-Jiun Ding

**Abstract**: Backdoor data poisoning is a crucial technique for ownership protection and defending against malicious attacks. Embedding hidden triggers in training data can manipulate model outputs, enabling provenance verification, and deterring unauthorized use. However, current audio backdoor methods are suboptimal, as poisoned audio often exhibits degraded perceptual quality, which is noticeable to human listeners. This work explores the intrinsic stealthiness and effectiveness of audio watermarking in achieving successful poisoning. We propose a novel Watermark-as-Trigger concept, integrated into the Bloodroot backdoor framework via adversarial LoRA fine-tuning, which enhances perceptual quality while achieving a much higher trigger success rate and clean-sample accuracy. Experiments on speech recognition (SR) and speaker identification (SID) datasets show that watermark-based poisoning remains effective under acoustic filtering and model pruning. The proposed Bloodroot backdoor framework not only secures data-to-model ownership, but also well reveals the risk of adversarial misuse.



## **12. AEGIS : Automated Co-Evolutionary Framework for Guarding Prompt Injections Schema**

cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.00088v2) [paper-pdf](http://arxiv.org/pdf/2509.00088v2)

**Authors**: Ting-Chun Liu, Ching-Yu Hsu, Kuan-Yi Lee, Chi-An Fu, Hung-yi Lee

**Abstract**: Prompt injection attacks pose a significant challenge to the safe deployment of Large Language Models (LLMs) in real-world applications. While prompt-based detection offers a lightweight and interpretable defense strategy, its effectiveness has been hindered by the need for manual prompt engineering. To address this issue, we propose AEGIS , an Automated co-Evolutionary framework for Guarding prompt Injections Schema. Both attack and defense prompts are iteratively optimized against each other using a gradient-like natural language prompt optimization technique. This framework enables both attackers and defenders to autonomously evolve via a Textual Gradient Optimization (TGO) module, leveraging feedback from an LLM-guided evaluation loop. We evaluate our system on a real-world assignment grading dataset of prompt injection attacks and demonstrate that our method consistently outperforms existing baselines, achieving superior robustness in both attack success and detection. Specifically, the attack success rate (ASR) reaches 1.0, representing an improvement of 0.26 over the baseline. For detection, the true positive rate (TPR) improves by 0.23 compared to the previous best work, reaching 0.84, and the true negative rate (TNR) remains comparable at 0.89. Ablation studies confirm the importance of co-evolution, gradient buffering, and multi-objective optimization. We also confirm that this framework is effective in different LLMs. Our results highlight the promise of adversarial training as a scalable and effective approach for guarding prompt injections.



## **13. Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs**

cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07697v1) [paper-pdf](http://arxiv.org/pdf/2510.07697v1)

**Authors**: Man Hu, Xinyi Wu, Zuofeng Suo, Jinbo Feng, Linghui Meng, Yanhao Jia, Anh Tuan Luu, Shuai Zhao

**Abstract**: With the rise of advanced reasoning capabilities, large language models (LLMs) are receiving increasing attention. However, although reasoning improves LLMs' performance on downstream tasks, it also introduces new security risks, as adversaries can exploit these capabilities to conduct backdoor attacks. Existing surveys on backdoor attacks and reasoning security offer comprehensive overviews but lack in-depth analysis of backdoor attacks and defenses targeting LLMs' reasoning abilities. In this paper, we take the first step toward providing a comprehensive review of reasoning-based backdoor attacks in LLMs by analyzing their underlying mechanisms, methodological frameworks, and unresolved challenges. Specifically, we introduce a new taxonomy that offers a unified perspective for summarizing existing approaches, categorizing reasoning-based backdoor attacks into associative, passive, and active. We also present defense strategies against such attacks and discuss current challenges alongside potential directions for future research. This work offers a novel perspective, paving the way for further exploration of secure and trustworthy LLM communities.



## **14. DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation with Uncertainty-Quantification Support**

cs.LG

18 pages, 9 figures, 5 tables

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07620v1) [paper-pdf](http://arxiv.org/pdf/2510.07620v1)

**Authors**: Muhammad Usman, Yugyung Lee

**Abstract**: Dynamic trust evaluation in large, rapidly evolving graphs requires models that can capture changing relationships, express calibrated confidence, and resist adversarial manipulation. DGTEN (Deep Gaussian-based Trust Evaluation Network) introduces a unified graph framework that achieves all three by combining uncertainty-aware message passing, expressive temporal modeling, and built-in defenses against trust-targeted attacks. It represents nodes and edges as Gaussian distributions so that both semantic signals and epistemic uncertainty propagate through the graph neural network, enabling risk-aware trust decisions rather than overconfident guesses. To model how trust evolves, it employs hybrid Absolute-Gaussian-Hourglass (HAGH) positional encoding with Kolmogorov-Arnold network-based unbiased multi-head attention, followed by an ordinary differential equation (ODE)-based residual learning module to jointly capture abrupt shifts and smooth trends. Robust adaptive ensemble coefficient analysis prunes or down-weights suspicious interactions using complementary cosine and Jaccard similarity measures, mitigating reputation laundering, sabotage, and on/off attacks. On two signed Bitcoin trust networks, DGTEN delivers significant improvements: in single-timeslot prediction on Bitcoin-Alpha, it improves MCC by 10.77% over the best dynamic baseline; in the cold-start scenario, it achieves a 16.41% MCC gain - the largest across all tasks and datasets. Under adversarial on/off attacks, it surpasses the baseline by up to 11.63% MCC. These results validate the effectiveness of the unified DGTEN framework.



## **15. $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks**

cs.MA

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2504.00218v2) [paper-pdf](http://arxiv.org/pdf/2504.00218v2)

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Fleming, Tianlong Chen

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.



## **16. MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification**

cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2406.05927v3) [paper-pdf](http://arxiv.org/pdf/2406.05927v3)

**Authors**: Sajjad Amini, Mohammadreza Teymoorianfard, Shiqing Ma, Amir Houmansadr

**Abstract**: We present a simple yet effective method to improve the robustness of both Convolutional and attention-based Neural Networks against adversarial examples by post-processing an adversarially trained model. Our technique, MeanSparse, cascades the activation functions of a trained model with novel operators that sparsify mean-centered feature vectors. This is equivalent to reducing feature variations around the mean, and we show that such reduced variations merely affect the model's utility, yet they strongly attenuate the adversarial perturbations and decrease the attacker's success rate. Our experiments show that, when applied to the top models in the RobustBench leaderboard, MeanSparse achieves a new robustness record of 75.28% (from 73.71%), 44.78% (from 42.67%) and 62.12% (from 59.56%) on CIFAR-10, CIFAR-100 and ImageNet, respectively, in terms of AutoAttack accuracy. Code is available at https://github.com/SPIN-UMass/MeanSparse



## **17. LLMs Encode Harmfulness and Refusal Separately**

cs.CL

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2507.11878v3) [paper-pdf](http://arxiv.org/pdf/2507.11878v3)

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.



## **18. D2RA: Dual Domain Regeneration Attack**

cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07538v1) [paper-pdf](http://arxiv.org/pdf/2510.07538v1)

**Authors**: Pragati Shuddhodhan Meshram, Varun Chandrasekaran

**Abstract**: The growing use of generative models has intensified the need for watermarking methods that ensure content attribution and provenance. While recent semantic watermarking schemes improve robustness by embedding signals in latent or frequency representations, we show they remain vulnerable even under resource-constrained adversarial settings. We present D2RA, a training-free, single-image attack that removes or weakens watermarks without access to the underlying model. By projecting watermarked images onto natural priors across complementary representations, D2RA suppresses watermark signals while preserving visual fidelity. Experiments across diverse watermarking schemes demonstrate that our approach consistently reduces watermark detectability, revealing fundamental weaknesses in current designs. Our code is available at https://github.com/Pragati-Meshram/DAWN.



## **19. PEAR: Planner-Executor Agent Robustness Benchmark**

cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07505v1) [paper-pdf](http://arxiv.org/pdf/2510.07505v1)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.



## **20. SpecGuard: Spectral Projection-based Advanced Invisible Watermarking**

cs.CV

ICCV 2025 Accepted Paper

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07302v1) [paper-pdf](http://arxiv.org/pdf/2510.07302v1)

**Authors**: Inzamamul Alam, Md Tanvir Islam, Khan Muhammad, Simon S. Woo

**Abstract**: Watermarking embeds imperceptible patterns into images for authenticity verification. However, existing methods often lack robustness against various transformations primarily including distortions, image regeneration, and adversarial perturbation, creating real-world challenges. In this work, we introduce SpecGuard, a novel watermarking approach for robust and invisible image watermarking. Unlike prior approaches, we embed the message inside hidden convolution layers by converting from the spatial domain to the frequency domain using spectral projection of a higher frequency band that is decomposed by wavelet projection. Spectral projection employs Fast Fourier Transform approximation to transform spatial data into the frequency domain efficiently. In the encoding phase, a strength factor enhances resilience against diverse attacks, including adversarial, geometric, and regeneration-based distortions, ensuring the preservation of copyrighted information. Meanwhile, the decoder leverages Parseval's theorem to effectively learn and extract the watermark pattern, enabling accurate retrieval under challenging transformations. We evaluate the proposed SpecGuard based on the embedded watermark's invisibility, capacity, and robustness. Comprehensive experiments demonstrate the proposed SpecGuard outperforms the state-of-the-art models. To ensure reproducibility, the full code is released on \href{https://github.com/inzamamulDU/SpecGuard_ICCV_2025}{\textcolor{blue}{\textbf{GitHub}}}.



## **21. L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)**

cs.AI

This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07363v1) [paper-pdf](http://arxiv.org/pdf/2510.07363v1)

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.



## **22. Differential Privacy for Adaptive Weight Aggregation in Federated Tumor Segmentation**

cs.LG

I have changed the methodology because of some technical errors in  this version

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2308.00856v2) [paper-pdf](http://arxiv.org/pdf/2308.00856v2)

**Authors**: Muhammad Irfan Khan, Esa Alhoniemi, Elina Kontio, Suleiman A. Khan, Mojtaba Jafaritadi

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that safeguards privacy by creating an impartial global model while respecting the privacy of individual client data. However, the conventional FL method can introduce security risks when dealing with diverse client data, potentially compromising privacy and data integrity. To address these challenges, we present a differential privacy (DP) federated deep learning framework in medical image segmentation. In this paper, we extend our similarity weight aggregation (SimAgg) method to DP-SimAgg algorithm, a differentially private similarity-weighted aggregation algorithm for brain tumor segmentation in multi-modal magnetic resonance imaging (MRI). Our DP-SimAgg method not only enhances model segmentation capabilities but also provides an additional layer of privacy preservation. Extensive benchmarking and evaluation of our framework, with computational performance as a key consideration, demonstrate that DP-SimAgg enables accurate and robust brain tumor segmentation while minimizing communication costs during model training. This advancement is crucial for preserving the privacy of medical image data and safeguarding sensitive information. In conclusion, adding a differential privacy layer in the global weight aggregation phase of the federated brain tumor segmentation provides a promising solution to privacy concerns without compromising segmentation model efficacy. By leveraging DP, we ensure the protection of client data against adversarial attacks and malicious participants.



## **23. Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples**

cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07192v1) [paper-pdf](http://arxiv.org/pdf/2510.07192v1)

**Authors**: Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies, Burak Hasircioglu, Ezzeldin Shereen, Carlos Mougan, Vasilios Mavroudis, Erik Jones, Chris Hicks, Nicholas Carlini, Yarin Gal, Robert Kirk

**Abstract**: Poisoning attacks can compromise the safety of large language models (LLMs) by injecting malicious documents into their training data. Existing work has studied pretraining poisoning assuming adversaries control a percentage of the training corpus. However, for large models, even small percentages translate to impractically large amounts of data. This work demonstrates for the first time that poisoning attacks instead require a near-constant number of documents regardless of dataset size. We conduct the largest pretraining poisoning experiments to date, pretraining models from 600M to 13B parameters on chinchilla-optimal datasets (6B to 260B tokens). We find that 250 poisoned documents similarly compromise models across all model and dataset sizes, despite the largest models training on more than 20 times more clean data. We also run smaller-scale experiments to ablate factors that could influence attack success, including broader ratios of poisoned to clean data and non-random distributions of poisoned samples. Finally, we demonstrate the same dynamics for poisoning during fine-tuning. Altogether, our results suggest that injecting backdoors through data poisoning may be easier for large models than previously believed as the number of poisons required does not scale up with model size, highlighting the need for more research on defences to mitigate this risk in future models.



## **24. Sustainable Self-evolution Adversarial Training**

cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2412.02270v2) [paper-pdf](http://arxiv.org/pdf/2412.02270v2)

**Authors**: Wenxuan Wang, Chenglei Wang, Huihui Qi, Menghao Ye, Xuelin Qian, Peng Wang, Yanning Zhang

**Abstract**: With the wide application of deep neural network models in various computer vision tasks, there has been a proliferation of adversarial example generation strategies aimed at deeply exploring model security. However, existing adversarial training defense models, which rely on single or limited types of attacks under a one-time learning process, struggle to adapt to the dynamic and evolving nature of attack methods. Therefore, to achieve defense performance improvements for models in long-term applications, we propose a novel Sustainable Self-Evolution Adversarial Training (SSEAT) framework. Specifically, we introduce a continual adversarial defense pipeline to realize learning from various kinds of adversarial examples across multiple stages. Additionally, to address the issue of model catastrophic forgetting caused by continual learning from ongoing novel attacks, we propose an adversarial data replay module to better select more diverse and key relearning data. Furthermore, we design a consistency regularization strategy to encourage current defense models to learn more from previously trained ones, guiding them to retain more past knowledge and maintain accuracy on clean samples. Extensive experiments have been conducted to verify the efficacy of the proposed SSEAT defense method, which demonstrates superior defense performance and classification accuracy compared to competitors.Code is available at https://github.com/aup520/SSEAT



## **25. Guardians of Image Quality: Benchmarking Defenses Against Adversarial Attacks on Image Quality Metrics**

cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2408.01541v2) [paper-pdf](http://arxiv.org/pdf/2408.01541v2)

**Authors**: Alexander Gushchin, Khaled Abud, Georgii Bychkov, Ekaterina Shumitskaya, Anna Chistyakova, Sergey Lavrushkin, Bader Rasheed, Kirill Malyshev, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: In the field of Image Quality Assessment (IQA), the adversarial robustness of the metrics poses a critical concern. This paper presents a comprehensive benchmarking study of various defense mechanisms in response to the rise in adversarial attacks on IQA. We systematically evaluate 25 defense strategies, including adversarial purification, adversarial training, and certified robustness methods. We applied 14 adversarial attack algorithms of various types in both non-adaptive and adaptive settings and tested these defenses against them. We analyze the differences between defenses and their applicability to IQA tasks, considering that they should preserve IQA scores and image quality. The proposed benchmark aims to guide future developments and accepts submissions of new methods, with the latest results available online: https://videoprocessing.ai/benchmarks/iqa-defenses.html.



## **26. Universally Composable Termination Analysis of Tendermint**

cs.CR

35 pages including references, 16 figures, 2 tables. Submitted to  ACNS 2026

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.01097v2) [paper-pdf](http://arxiv.org/pdf/2510.01097v2)

**Authors**: Zhixin Dong, Xian Xu, Yuhang Zeng, Mingchao Wan, Chunmiao Li

**Abstract**: Modern blockchain systems operating in adversarial environments require robust consensus protocols that guarantee both safety and termination under network delay attacks. Tendermint, a widely adopted consensus protocol in consortium blockchains, achieves high throughput and finality. However, previous analysis of the safety and termination has been done in a standalone fashion, with no consideration of the composition with other protocols interacting with it in a concurrent manner. Moreover, the termination properties under adaptive network delays caused by Byzantine adversaries have not been formally analyzed. This paper presents the first universally composable (UC) security analysis of Tendermint, demonstrating its resilience against strategic message-delay attacks. By constructing a UC ideal model of Tendermint, we formalize its core mechanisms: phase-base consensus procedure, dynamic timeouts, proposal locking, leader rotation, and others, under a network adversary that selectively delays protocol messages. Our main result proves that the Tendermint protocol UC-realizes the ideal Tendermint model, which ensures bounded termination latency, i.e., guaranteed termination, even when up to $f<n/3$ nodes are Byzantine (where $n$ is the number of nodes participating in the consensus), provided that network delays remain within a protocol-defined threshold under the partially synchronous net assumption. Specifically, through formal proofs within the UC framework, we show that Tendermint maintains safety and termination. By the composition theorem of UC, this guarantees that these properties are maintained when Tendermint is composed with various blockchain components.



## **27. DiffMI: Breaking Face Recognition Privacy via Diffusion-Driven Training-Free Model Inversion**

cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2504.18015v3) [paper-pdf](http://arxiv.org/pdf/2504.18015v3)

**Authors**: Hanrui Wang, Shuo Wang, Chun-Shien Lu, Isao Echizen

**Abstract**: Face recognition poses serious privacy risks due to its reliance on sensitive and immutable biometric data. While modern systems mitigate privacy risks by mapping facial images to embeddings (commonly regarded as privacy-preserving), model inversion attacks reveal that identity information can still be recovered, exposing critical vulnerabilities. However, existing attacks are often computationally expensive and lack generalization, especially those requiring target-specific training. Even training-free approaches suffer from limited identity controllability, hindering faithful reconstruction of nuanced or unseen identities. In this work, we propose DiffMI, the first diffusion-driven, training-free model inversion attack. DiffMI introduces a novel pipeline combining robust latent code initialization, a ranked adversarial refinement strategy, and a statistically grounded, confidence-aware optimization objective. DiffMI applies directly to unseen target identities and face recognition models, offering greater adaptability than training-dependent approaches while significantly reducing computational overhead. Our method achieves 84.42%--92.87% attack success rates against inversion-resilient systems and outperforms the best prior training-free GAN-based approach by 4.01%--9.82%. The implementation is available at https://github.com/azrealwang/DiffMI.



## **28. Minimal Cascade Gradient Smoothing for Fast Transferable Preemptive Adversarial Defense**

cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2407.15524v8) [paper-pdf](http://arxiv.org/pdf/2407.15524v8)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Ching-Chia Kao, Isao Echizen

**Abstract**: Adversarial attacks persist as a major challenge in deep learning. While training- and test-time defenses are well-studied, they often reduce clean accuracy, incur high cost, or fail under adaptive threats. In contrast, preemptive defenses, which perturb media before release, offer a practical alternative but remain slow, model-coupled, and brittle. We propose the Minimal Sufficient Preemptive Defense (MSPD), a fast, transferable framework that defends against future attacks without access to the target model or gradients. MSPD is driven by Minimal Cascade Gradient Smoothing (MCGS), a two-epoch optimization paradigm executed on a surrogate backbone. This defines a minimal yet effective regime for robust generalization across unseen models and attacks. MSPD runs at 0.02s/image (CIFAR-10) and 0.26s/image (ImageNet), 28--1696x faster than prior preemptive methods, while improving robust accuracy by +5% and clean accuracy by +3.7% across 11 models and 7 attacks. To evaluate adaptive robustness, we introduce Preemptive Reversion, the first white-box diagnostic attack that cancels preemptive perturbations under full gradient access. Even in this setting, MSPD retains a +2.2% robustness margin over the baseline. In practice, when gradients are unavailable, MSPD remains reliable and efficient. MSPD, MCGS, and Preemptive Reversion are each supported by formal theoretical proofs. The implementation is available at https://github.com/azrealwang/MSPD.



## **29. GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm**

cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.14230v2) [paper-pdf](http://arxiv.org/pdf/2501.14230v2)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Christopher Leckie, Isao Echizen

**Abstract**: Deep neural networks are highly vulnerable to adversarial examples that inputs with small, carefully crafted perturbations that cause misclassification, making adversarial attacks an essential tool for robustness evaluation. Existing black-box attacks fall into three categories: query-only, transfer-only, and query-and-transfer, and vary in perturbation pattern and optimization strategy. However, no prior method jointly achieves query-and-transfer guidance, pixel-wise sparsity, and training-free direct optimization, leaving a gap between black-box flexibility and white-box precision. We present GreedyPixel, a new attack framework that fills this gap by combining a surrogate-derived pixel priority map with greedy, per-pixel optimization refined by query feedback. This design reduces the exponential brute-force search space to a tractable linear procedure, guarantees monotonic loss decrease and convergence to a coordinate-wise optimum, and concentrates perturbations on robust, semantically meaningful pixels to improve perceptual quality. Extensive experiments on CIFAR-10 and ImageNet under both white-box and black-box settings demonstrate that GreedyPixel achieves state-of-the-art attack success rates and produces visually imperceptible perturbations. Our results show that GreedyPixel bridges the precision gap between white-box and black-box attacks and provides a practical framework for fine-grained robustness evaluation. The implementation is available at https://github.com/azrealwang/greedypixel.



## **30. RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning**

cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06994v1) [paper-pdf](http://arxiv.org/pdf/2510.06994v1)

**Authors**: Artur Horal, Daniel Pina, Henrique Paz, Iago Paulo, João Soares, Rafael Ferreira, Diogo Tavares, Diogo Glória-Silva, João Magalhães, David Semedo

**Abstract**: This paper presents the vision, scientific contributions, and technical details of RedTWIZ: an adaptive and diverse multi-turn red teaming framework, to audit the robustness of Large Language Models (LLMs) in AI-assisted software development. Our work is driven by three major research streams: (1) robust and systematic assessment of LLM conversational jailbreaks; (2) a diverse generative multi-turn attack suite, supporting compositional, realistic and goal-oriented jailbreak conversational strategies; and (3) a hierarchical attack planner, which adaptively plans, serializes, and triggers attacks tailored to specific LLM's vulnerabilities. Together, these contributions form a unified framework -- combining assessment, attack generation, and strategic planning -- to comprehensively evaluate and expose weaknesses in LLMs' robustness. Extensive evaluation is conducted to systematically assess and analyze the performance of the overall system and each component. Experimental results demonstrate that our multi-turn adversarial attack strategies can successfully lead state-of-the-art LLMs to produce unsafe generations, highlighting the pressing need for more research into enhancing LLM's robustness.



## **31. OBJVanish: Physically Realizable Text-to-3D Adv. Generation of LiDAR-Invisible Objects**

cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06952v1) [paper-pdf](http://arxiv.org/pdf/2510.06952v1)

**Authors**: Bing Li, Wuqi Wang, Yanan Zhang, Jingzheng Li, Haigen Min, Wei Feng, Xingyu Zhao, Jie Zhang, Qing Guo

**Abstract**: LiDAR-based 3D object detectors are fundamental to autonomous driving, where failing to detect objects poses severe safety risks. Developing effective 3D adversarial attacks is essential for thoroughly testing these detection systems and exposing their vulnerabilities before real-world deployment. However, existing adversarial attacks that add optimized perturbations to 3D points have two critical limitations: they rarely cause complete object disappearance and prove difficult to implement in physical environments. We introduce the text-to-3D adversarial generation method, a novel approach enabling physically realizable attacks that can generate 3D models of objects truly invisible to LiDAR detectors and be easily realized in the real world. Specifically, we present the first empirical study that systematically investigates the factors influencing detection vulnerability by manipulating the topology, connectivity, and intensity of individual pedestrian 3D models and combining pedestrians with multiple objects within the CARLA simulation environment. Building on the insights, we propose the physically-informed text-to-3D adversarial generation (Phy3DAdvGen) that systematically optimizes text prompts by iteratively refining verbs, objects, and poses to produce LiDAR-invisible pedestrians. To ensure physical realizability, we construct a comprehensive object pool containing 13 3D models of real objects and constrain Phy3DAdvGen to generate 3D objects based on combinations of objects in this set. Extensive experiments demonstrate that our approach can generate 3D pedestrians that evade six state-of-the-art (SOTA) LiDAR 3D detectors in both CARLA simulation and physical environments, thereby highlighting vulnerabilities in safety-critical applications.



## **32. Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness**

cs.LG

17 pages

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06790v1) [paper-pdf](http://arxiv.org/pdf/2510.06790v1)

**Authors**: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

**Abstract**: Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization, while RL finetuning and protracted reasoning are not critical. For example, increasing emphasis on defensive specifications via prompting lowers the success rate of gradient-based multimodal attacks on VLMs robustified by adversarial pretraining, but this same intervention provides no such benefit to not-robustified models. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Accordingly, we advise layering train-time and test-time defenses to obtain their synergistic benefit.



## **33. Benchmarking Gaslighting Negation Attacks Against Multimodal Large Language Models**

cs.CL

Project website:  https://yxg1005.github.io/GaslightingNegationAttacks/

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.19017v4) [paper-pdf](http://arxiv.org/pdf/2501.19017v4)

**Authors**: Bin Zhu, Yinxuan Gui, Huiyan Qi, Jingjing Chen, Chong-Wah Ngo, Ee-Peng Lim

**Abstract**: Multimodal Large Language Models (MLLMs) have exhibited remarkable advancements in integrating different modalities, excelling in complex understanding and generation tasks. Despite their success, MLLMs remain vulnerable to conversational adversarial inputs. In this paper, we systematically study gaslighting negation attacks: a phenomenon where models, despite initially providing correct answers, are persuaded by user-provided negations to reverse their outputs, often fabricating justifications. We conduct extensive evaluations of state-of-the-art MLLMs across diverse benchmarks and observe substantial performance drops when negation is introduced. Notably, we introduce the first benchmark GaslightingBench, specifically designed to evaluate the vulnerability of MLLMs to negation arguments. GaslightingBench consists of multiple-choice questions curated from existing datasets, along with generated negation prompts across 20 diverse categories. Throughout extensive evaluation, we find that proprietary models such as Gemini-1.5-flash and GPT-4o demonstrate better resilience compared to open-source counterparts like Qwen2-VL and LLaVA, though even advanced reasoning-oriented models like Gemini-2.5-Pro remain susceptible. Our category-level analysis further shows that subjective or socially nuanced domains (e.g., Social Relation, Image Emotion) are especially fragile, while more objective domains (e.g., Geography) exhibit relatively smaller but still notable drops. Overall, all evaluated MLLMs struggle to maintain logical consistency under gaslighting negation attack. These findings highlight a fundamental robustness gap and provide insights for developing more reliable and trustworthy multimodal AI systems. Project website: https://yxg1005.github.io/GaslightingNegationAttacks/.



## **34. Towards the Worst-case Robustness of Large Language Models**

cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.19040v4) [paper-pdf](http://arxiv.org/pdf/2501.19040v4)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.



## **35. SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models**

cs.CR

Accepted by ACM CCS 2025

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.05173v2) [paper-pdf](http://arxiv.org/pdf/2510.05173v2)

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce \textbf{SafeGuider}, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, \textbf{SafeGuider} generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.



## **36. Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?**

cs.CL

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06594v1) [paper-pdf](http://arxiv.org/pdf/2510.06594v1)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.



## **37. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2502.17832v3) [paper-pdf](http://arxiv.org/pdf/2502.17832v3)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-Wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models with Retrieval Augmented Generation (RAG) have significantly advanced tasks such as multimodal question answering by grounding responses in external text and images. This grounding improves factuality, reduces hallucination, and extends reasoning beyond parametric knowledge. However, this reliance on external knowledge poses a critical yet underexplored safety risk: knowledge poisoning attacks, where adversaries deliberately inject adversarial multimodal content into external knowledge bases to steer model toward generating incorrect or even harmful responses. To expose such vulnerabilities, we propose MM-PoisonRAG, the first framework to systematically design knowledge poisoning in multimodal RAG. We introduce two complementary attack strategies: Localized Poisoning Attack (LPA), which implants targeted multimodal misinformation to manipulate specific queries, and Globalized Poisoning Attack (GPA), which inserts a single adversarial knowledge to broadly disrupt reasoning and induce nonsensical responses across all queries. Comprehensive experiments across tasks, models, and access settings show that LPA achieves targeted manipulation with attack success rates of up to 56%, while GPA completely disrupts model generation to 0% accuracy with just a single adversarial knowledge injection. Our results reveal the fragility of multimodal RAG and highlight the urgent need for defenses against knowledge poisoning.



## **38. Text-to-Image Models Leave Identifiable Signatures: Implications for Leaderboard Security**

cs.LG

Accepted at Lock-LLM Workshop, NeurIPS 2025

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.06525v1) [paper-pdf](http://arxiv.org/pdf/2510.06525v1)

**Authors**: Ali Naseh, Anshuman Suri, Yuefeng Peng, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Generative AI leaderboards are central to evaluating model capabilities, but remain vulnerable to manipulation. Among key adversarial objectives is rank manipulation, where an attacker must first deanonymize the models behind displayed outputs -- a threat previously demonstrated and explored for large language models (LLMs). We show that this problem can be even more severe for text-to-image leaderboards, where deanonymization is markedly easier. Using over 150,000 generated images from 280 prompts and 19 diverse models spanning multiple organizations, architectures, and sizes, we demonstrate that simple real-time classification in CLIP embedding space identifies the generating model with high accuracy, even without prompt control or historical data. We further introduce a prompt-level separability metric and identify prompts that enable near-perfect deanonymization. Our results indicate that rank manipulation in text-to-image leaderboards is easier than previously recognized, underscoring the need for stronger defenses.



## **39. Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples**

cs.NE

Accepted manuscript. Published in *Neurocomputing*, Volume 656, 2025,  Article 131506. Available online 12 September 2025. DOI:  10.1016/j.neucom.2025.131506

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2209.03358v4) [paper-pdf](http://arxiv.org/pdf/2209.03358v4)

**Authors**: Nuo Xu, Kaleel Mahmood, Haowen Fang, Ethan Rathbun, Caiwen Ding, Wujie Wen

**Abstract**: Spiking neural networks (SNNs) have drawn much attention for their high energy efficiency and recent advances in classification performance. However, unlike traditional deep learning, the robustness of SNNs to adversarial examples remains underexplored. This work advances the adversarial attack side of SNNs and makes three major contributions. First, we show that successful white-box attacks on SNNs strongly depend on the surrogate gradient estimation technique, even for adversarially trained models. Second, using the best single surrogate gradient estimator, we study the transferability of adversarial examples between SNNs and state-of-the-art architectures such as Vision Transformers (ViTs) and CNNs. Our analysis reveals two major gaps: no existing white-box attack leverages multiple surrogate estimators, and no single attack effectively fools both SNNs and non-SNN models simultaneously. Third, we propose the Mixed Dynamic Spiking Estimation (MDSE) attack, which dynamically combines multiple surrogate gradients to overcome these gaps. MDSE produces adversarial examples that fool both SNN and non-SNN models, achieving up to 91.4% higher effectiveness on SNN/ViT ensembles and a 3x boost on adversarially trained SNN ensembles over Auto-PGD. Experiments span three datasets (CIFAR-10, CIFAR-100, ImageNet) and nineteen classifiers, and we will release code and models upon publication.



## **40. Adversarial Surrogate Risk Bounds for Binary Classification**

cs.LG

37 pages, 3 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2506.09348v2) [paper-pdf](http://arxiv.org/pdf/2506.09348v2)

**Authors**: Natalie S. Frank

**Abstract**: A central concern in classification is the vulnerability of machine learning models to adversarial attacks. Adversarial training is one of the most popular techniques for training robust classifiers, which involves minimizing an adversarial surrogate risk. Recent work has characterized the conditions under which any sequence minimizing the adversarial surrogate risk also minimizes the adversarial classification risk in the binary setting, a property known as adversarial consistency. However, these results do not address the rate at which the adversarial classification risk approaches its optimal value along such a sequence. This paper provides surrogate risk bounds that quantify that convergence rate.



## **41. LLM Unlearning via Neural Activation Redirection**

cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2502.07218v2) [paper-pdf](http://arxiv.org/pdf/2502.07218v2)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: The ability to selectively remove knowledge from LLMs is highly desirable. However, existing methods often struggle with balancing unlearning efficacy and retain model utility, and lack controllability at inference time to emulate base model behavior as if it had never seen the unlearned data. In this paper, we propose LUNAR, a novel unlearning method grounded in the Linear Representation Hypothesis and operates by redirecting the representations of unlearned data to activation regions that expresses its inability to answer. We show that contrastive features are not a prerequisite for effective activation redirection, and LUNAR achieves state-of-the-art unlearning performance and superior controllability. Specifically, LUNAR achieves between 2.9x and 11.7x improvement in the combined unlearning efficacy and model utility score (Deviation Score) across various base models and generates coherent, contextually appropriate responses post-unlearning. Moreover, LUNAR effectively reduces parameter updates to a single down-projection matrix, a novel design that significantly enhances efficiency by 20x and robustness. Finally, we demonstrate that LUNAR is robust to white-box adversarial attacks and versatile in real-world scenarios, including handling sequential unlearning requests.



## **42. Breaking Precision Time: OS Vulnerability Exploits Against IEEE 1588**

cs.CR

Published in IEEE ISPCS 2025

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.06421v1) [paper-pdf](http://arxiv.org/pdf/2510.06421v1)

**Authors**: Muhammad Abdullah Soomro, Fatima Muhammad Anwar

**Abstract**: The Precision Time Protocol (PTP), standardized as IEEE 1588, provides sub-microsecond synchronization across distributed systems and underpins critical infrastructure in telecommunications, finance, power systems, and industrial automation. While prior work has extensively analyzed PTP's vulnerability to network-based attacks, prompting the development of cryptographic protections and anomaly detectors, these defenses presume an uncompromised host. In this paper, we identify and exploit a critical blind spot in current threat models: kernel-level adversaries operating from within the host running the PTP stack. We present the first systematic study of kernel-rooted attacks on PTP, demonstrating how privileged attackers can manipulate system time by corrupting key interfaces without altering PTP network traffic. We implement three attack primitives, constant offset, progressive skew, and random jitter, using in-kernel payloads, and evaluate their impact on the widely used ptp4l and phc2sys daemons. Our experiments reveal that these attacks can silently destabilize clock synchronization, bypassing existing PTP security extensions. These findings highlight the urgent need to reconsider host-level trust assumptions and integrate kernel integrity into the design of secure time synchronization systems.



## **43. When Should Selfish Miners Double-Spend?**

cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2501.03227v2) [paper-pdf](http://arxiv.org/pdf/2501.03227v2)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.



## **44. Sparse Representations Improve Adversarial Robustness of Neural Network Classifiers**

cs.LG

Killian Steunou is the main contributor and corresponding author of  this work

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2509.21130v2) [paper-pdf](http://arxiv.org/pdf/2509.21130v2)

**Authors**: Killian Steunou, Théo Druilhe, Sigurd Saue

**Abstract**: Deep neural networks perform remarkably well on image classification tasks but remain vulnerable to carefully crafted adversarial perturbations. This work revisits linear dimensionality reduction as a simple, data-adapted defense. We empirically compare standard Principal Component Analysis (PCA) with its sparse variant (SPCA) as front-end feature extractors for downstream classifiers, and we complement these experiments with a theoretical analysis. On the theory side, we derive exact robustness certificates for linear heads applied to SPCA features: for both $\ell_\infty$ and $\ell_2$ threat models (binary and multiclass), the certified radius grows as the dual norms of $W^\top u$ shrink, where $W$ is the projection and $u$ the head weights. We further show that for general (non-linear) heads, sparsity reduces operator-norm bounds through a Lipschitz composition argument, predicting lower input sensitivity. Empirically, with a small non-linear network after the projection, SPCA consistently degrades more gracefully than PCA under strong white-box and black-box attacks while maintaining competitive clean accuracy. Taken together, the theory identifies the mechanism (sparser projections reduce adversarial leverage) and the experiments verify that this benefit persists beyond the linear setting. Our code is available at https://github.com/killian31/SPCARobustness.



## **45. SAFER: Advancing Safety Alignment via Efficient Ex-Ante Reasoning**

cs.CL

22 pages, 5 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2504.02725v2) [paper-pdf](http://arxiv.org/pdf/2504.02725v2)

**Authors**: Kehua Feng, Keyan Ding, Yuhao Wang, Menghan Li, Fanjunduo Wei, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose SAFER, a framework for Safety Alignment via eFficient Ex-Ante Reasoning. Our approach instantiates structured Ex-Ante reasoning through initial assessment, rule verification, and path calibration, and embeds predefined safety rules to provide transparent and verifiable safety judgments. Specifically, our approach consists of two training stages: (1) supervised fine-tuning with synthetic traces to teach the multi-stage Ex-Ante reasoning, and (2) step-level reasoning preference optimization to jointly enhance safety, utility, and efficiency. Experiments on multiple open-source LLMs demonstrate that SAFER significantly enhances safety performance while maintaining helpfulness and response efficiency.



## **46. DP-SNP-TIHMM: Differentially Private, Time-Inhomogeneous Hidden Markov Models for Synthesizing Genome-Wide Association Datasets**

cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05777v1) [paper-pdf](http://arxiv.org/pdf/2510.05777v1)

**Authors**: Shadi Rahimian, Mario Fritz

**Abstract**: Single nucleotide polymorphism (SNP) datasets are fundamental to genetic studies but pose significant privacy risks when shared. The correlation of SNPs with each other makes strong adversarial attacks such as masked-value reconstruction, kin, and membership inference attacks possible. Existing privacy-preserving approaches either apply differential privacy to statistical summaries of these datasets or offer complex methods that require post-processing and the usage of a publicly available dataset to suppress or selectively share SNPs.   In this study, we introduce an innovative framework for generating synthetic SNP sequence datasets using samples derived from time-inhomogeneous hidden Markov models (TIHMMs). To preserve the privacy of the training data, we ensure that each SNP sequence contributes only a bounded influence during training, enabling strong differential privacy guarantees. Crucially, by operating on full SNP sequences and bounding their gradient contributions, our method directly addresses the privacy risks introduced by their inherent correlations.   Through experiments conducted on the real-world 1000 Genomes dataset, we demonstrate the efficacy of our method using privacy budgets of $\varepsilon \in [1, 10]$ at $\delta=10^{-4}$. Notably, by allowing the transition models of the HMM to be dependent on the location in the sequence, we significantly enhance performance, enabling the synthetic datasets to closely replicate the statistical properties of non-private datasets. This framework facilitates the private sharing of genomic data while offering researchers exceptional flexibility and utility.



## **47. Evidence of Cognitive Biases in Capture-the-Flag Cybersecurity Competitions**

cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05771v1) [paper-pdf](http://arxiv.org/pdf/2510.05771v1)

**Authors**: Carolina Carreira, Anu Aggarwal, Alejandro Cuevas, Maria José Ferreira, Hanan Hibshi, Cleotilde Gonzalez

**Abstract**: Understanding how cognitive biases influence adversarial decision-making is essential for developing effective cyber defenses. Capture-the-Flag (CTF) competitions provide an ecologically valid testbed to study attacker behavior at scale, simulating real-world intrusion scenarios under pressure. We analyze over 500,000 submission logs from picoCTF, a large educational CTF platform, to identify behavioral signatures of cognitive biases with defensive implications. Focusing on availability bias and the sunk cost fallacy, we employ a mixed-methods approach combining qualitative coding, descriptive statistics, and generalized linear modeling. Our findings show that participants often submitted flags with correct content but incorrect formatting (availability bias), and persisted in attempting challenges despite repeated failures and declining success probabilities (sunk cost fallacy). These patterns reveal that biases naturally shape attacker behavior in adversarial contexts. Building on these insights, we outline a framework for bias-informed adaptive defenses that anticipate, rather than simply react to, adversarial actions.



## **48. Shortcuts Everywhere and Nowhere: Exploring Multi-Trigger Backdoor Attacks**

cs.LG

13 pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2401.15295v4) [paper-pdf](http://arxiv.org/pdf/2401.15295v4)

**Authors**: Yige Li, Jiabo He, Hanxun Huang, Jun Sun, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks have become a significant threat to the pre-training and deployment of deep neural networks (DNNs). Although numerous methods for detecting and mitigating backdoor attacks have been proposed, most rely on identifying and eliminating the ``shortcut" created by the backdoor, which links a specific source class to a target class. However, these approaches can be easily circumvented by designing multiple backdoor triggers that create shortcuts everywhere and therefore nowhere specific. In this study, we explore the concept of Multi-Trigger Backdoor Attacks (MTBAs), where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks including \textit{parallel}, \textit{sequential}, and \textit{hybrid} attacks, we demonstrate that 1) multiple triggers can coexist, overwrite, or cross-activate one another, and 2) MTBAs easily break the prevalent shortcut assumption underlying most existing backdoor detection/removal methods, rendering them ineffective. Given the security risk posed by MTBAs, we have created a multi-trigger backdoor poisoning dataset to facilitate future research on detecting and mitigating these attacks, and we also discuss potential defense strategies against MTBAs. Our code is available at https://github.com/bboylyg/Multi-Trigger-Backdoor-Attacks.



## **49. Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension**

cs.CL

40 Pages, 6 figues

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2503.03502v2) [paper-pdf](http://arxiv.org/pdf/2503.03502v2)

**Authors**: Canaan Yung, Hanxun Huang, Christopher Leckie, Sarah Erfani

**Abstract**: Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours, posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood. In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces. Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalises across multiple LLMs and attack families.



## **50. Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms**

cs.LG

54 Pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2508.16481v2) [paper-pdf](http://arxiv.org/pdf/2508.16481v2)

**Authors**: Jonathan Nöther, Adish Singla, Goran Radanovic

**Abstract**: Ensuring the safe use of agentic systems requires a thorough understanding of the range of malicious behaviors these systems may exhibit when under attack. In this paper, we evaluate the robustness of LLM-based agentic systems against attacks that aim to elicit harmful actions from agents. To this end, we propose a novel taxonomy of harms for agentic systems and a novel benchmark, BAD-ACTS, for studying the security of agentic systems with respect to a wide range of harmful actions. BAD-ACTS consists of 4 implementations of agentic systems in distinct application environments, as well as a dataset of 188 high-quality examples of harmful actions. This enables a comprehensive study of the robustness of agentic systems across a wide range of categories of harmful behaviors, available tools, and inter-agent communication structures. Using this benchmark, we analyze the robustness of agentic systems against an attacker that controls one of the agents in the system and aims to manipulate other agents to execute a harmful target action. Our results show that the attack has a high success rate, demonstrating that even a single adversarial agent within the system can have a significant impact on the security. This attack remains effective even when agents use a simple prompting-based defense strategy. However, we additionally propose a more effective defense based on message monitoring. We believe that this benchmark provides a diverse testbed for the security research of agentic systems. The benchmark can be found at github.com/JNoether/BAD-ACTS



