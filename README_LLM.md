# Latest Large Language Model Attack Papers
**update at 2025-06-18 11:05:27**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. AIRTBench: Measuring Autonomous AI Red Teaming Capabilities in Language Models**

cs.CR

43 pages, 13 figures, 16 tables

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14682v1) [paper-pdf](http://arxiv.org/pdf/2506.14682v1)

**Authors**: Ads Dawson, Rob Mulla, Nick Landers, Shane Caldwell

**Abstract**: We introduce AIRTBench, an AI red teaming benchmark for evaluating language models' ability to autonomously discover and exploit Artificial Intelligence and Machine Learning (AI/ML) security vulnerabilities. The benchmark consists of 70 realistic black-box capture-the-flag (CTF) challenges from the Crucible challenge environment on the Dreadnode platform, requiring models to write python code to interact with and compromise AI systems. Claude-3.7-Sonnet emerged as the clear leader, solving 43 challenges (61% of the total suite, 46.9% overall success rate), with Gemini-2.5-Pro following at 39 challenges (56%, 34.3% overall), GPT-4.5-Preview at 34 challenges (49%, 36.9% overall), and DeepSeek R1 at 29 challenges (41%, 26.9% overall). Our evaluations show frontier models excel at prompt injection attacks (averaging 49% success rates) but struggle with system exploitation and model inversion challenges (below 26%, even for the best performers). Frontier models are far outpacing open-source alternatives, with the best truly open-source model (Llama-4-17B) solving 7 challenges (10%, 1.0% overall), though demonstrating specialized capabilities on certain hard challenges. Compared to human security researchers, large language models (LLMs) solve challenges with remarkable efficiency completing in minutes what typically takes humans hours or days-with efficiency advantages of over 5,000x on hard challenges. Our contribution fills a critical gap in the evaluation landscape, providing the first comprehensive benchmark specifically designed to measure and track progress in autonomous AI red teaming capabilities.



## **2. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.12442v3) [paper-pdf](http://arxiv.org/pdf/2505.12442v3)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.



## **3. Doppelgänger Method: Breaking Role Consistency in LLM Agent via Prompt-based Transferable Adversarial Attack**

cs.AI

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14539v1) [paper-pdf](http://arxiv.org/pdf/2506.14539v1)

**Authors**: Daewon Kang, YeongHwan Shin, Doyeon Kim, Kyu-Hwan Jung, Meong Hi Son

**Abstract**: Since the advent of large language models, prompt engineering now enables the rapid, low-effort creation of diverse autonomous agents that are already in widespread use. Yet this convenience raises urgent concerns about the safety, robustness, and behavioral consistency of the underlying prompts, along with the pressing challenge of preventing those prompts from being exposed to user's attempts. In this paper, we propose the ''Doppelg\"anger method'' to demonstrate the risk of an agent being hijacked, thereby exposing system instructions and internal information. Next, we define the ''Prompt Alignment Collapse under Adversarial Transfer (PACAT)'' level to evaluate the vulnerability to this adversarial transfer attack. We also propose a ''Caution for Adversarial Transfer (CAT)'' prompt to counter the Doppelg\"anger method. The experimental results demonstrate that the Doppelg\"anger method can compromise the agent's consistency and expose its internal information. In contrast, CAT prompts enable effective defense against this adversarial attack.



## **4. LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops**

cs.CL

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14493v1) [paper-pdf](http://arxiv.org/pdf/2506.14493v1)

**Authors**: Jiyuan Fu, Kaixun Jiang, Lingyi Hong, Jinglun Li, Haijing Guo, Dingkang Yang, Zhaoyu Chen, Wenqiang Zhang

**Abstract**: Multimodal Large Language Models (MLLMs) have shown great promise but require substantial computational resources during inference. Attackers can exploit this by inducing excessive output, leading to resource exhaustion and service degradation. Prior energy-latency attacks aim to increase generation time by broadly shifting the output token distribution away from the EOS token, but they neglect the influence of token-level Part-of-Speech (POS) characteristics on EOS and sentence-level structural patterns on output counts, limiting their efficacy. To address this, we propose LingoLoop, an attack designed to induce MLLMs to generate excessively verbose and repetitive sequences. First, we find that the POS tag of a token strongly affects the likelihood of generating an EOS token. Based on this insight, we propose a POS-Aware Delay Mechanism to postpone EOS token generation by adjusting attention weights guided by POS information. Second, we identify that constraining output diversity to induce repetitive loops is effective for sustained generation. We introduce a Generative Path Pruning Mechanism that limits the magnitude of hidden states, encouraging the model to produce persistent loops. Extensive experiments demonstrate LingoLoop can increase generated tokens by up to 30 times and energy consumption by a comparable factor on models like Qwen2.5-VL-3B, consistently driving MLLMs towards their maximum generation limits. These findings expose significant MLLMs' vulnerabilities, posing challenges for their reliable deployment. The code will be released publicly following the paper's acceptance.



## **5. Ensemble Watermarks for Large Language Models**

cs.CL

Accepted to ACL 2025 main conference. This article extends our  earlier work arXiv:2405.08400 by introducing an ensemble of stylometric  watermarking features and alternative experimental analysis. Code and data  are available at http://github.com/CommodoreEU/ensemble-watermark

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2411.19563v2) [paper-pdf](http://arxiv.org/pdf/2411.19563v2)

**Authors**: Georg Niess, Roman Kern

**Abstract**: As large language models (LLMs) reach human-like fluency, reliably distinguishing AI-generated text from human authorship becomes increasingly difficult. While watermarks already exist for LLMs, they often lack flexibility and struggle with attacks such as paraphrasing. To address these issues, we propose a multi-feature method for generating watermarks that combines multiple distinct watermark features into an ensemble watermark. Concretely, we combine acrostica and sensorimotor norms with the established red-green watermark to achieve a 98% detection rate. After a paraphrasing attack, the performance remains high with 95% detection rate. In comparison, the red-green feature alone as a baseline achieves a detection rate of 49% after paraphrasing. The evaluation of all feature combinations reveals that the ensemble of all three consistently has the highest detection rate across several LLMs and watermark strength settings. Due to the flexibility of combining features in the ensemble, various requirements and trade-offs can be addressed. Additionally, the same detection function can be used without adaptations for all ensemble configurations. This method is particularly of interest to facilitate accountability and prevent societal harm.



## **6. Excessive Reasoning Attack on Reasoning LLMs**

cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14374v1) [paper-pdf](http://arxiv.org/pdf/2506.14374v1)

**Authors**: Wai Man Si, Mingjie Li, Michael Backes, Yang Zhang

**Abstract**: Recent reasoning large language models (LLMs), such as OpenAI o1 and DeepSeek-R1, exhibit strong performance on complex tasks through test-time inference scaling. However, prior studies have shown that these models often incur significant computational costs due to excessive reasoning, such as frequent switching between reasoning trajectories (e.g., underthinking) or redundant reasoning on simple questions (e.g., overthinking). In this work, we expose a novel threat: adversarial inputs can be crafted to exploit excessive reasoning behaviors and substantially increase computational overhead without compromising model utility. Therefore, we propose a novel loss framework consisting of three components: (1) Priority Cross-Entropy Loss, a modification of the standard cross-entropy objective that emphasizes key tokens by leveraging the autoregressive nature of LMs; (2) Excessive Reasoning Loss, which encourages the model to initiate additional reasoning paths during inference; and (3) Delayed Termination Loss, which is designed to extend the reasoning process and defer the generation of final outputs. We optimize and evaluate our attack for the GSM8K and ORCA datasets on DeepSeek-R1-Distill-LLaMA and DeepSeek-R1-Distill-Qwen. Empirical results demonstrate a 3x to 9x increase in reasoning length with comparable utility performance. Furthermore, our crafted adversarial inputs exhibit transferability, inducing computational overhead in o3-mini, o1-mini, DeepSeek-R1, and QWQ models.



## **7. LLM-Powered Intent-Based Categorization of Phishing Emails**

cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14337v1) [paper-pdf](http://arxiv.org/pdf/2506.14337v1)

**Authors**: Even Eilertsen, Vasileios Mavroeidis, Gudmund Grov

**Abstract**: Phishing attacks remain a significant threat to modern cybersecurity, as they successfully deceive both humans and the defense mechanisms intended to protect them. Traditional detection systems primarily focus on email metadata that users cannot see in their inboxes. Additionally, these systems struggle with phishing emails, which experienced users can often identify empirically by the text alone. This paper investigates the practical potential of Large Language Models (LLMs) to detect these emails by focusing on their intent. In addition to the binary classification of phishing emails, the paper introduces an intent-type taxonomy, which is operationalized by the LLMs to classify emails into distinct categories and, therefore, generate actionable threat information. To facilitate our work, we have curated publicly available datasets into a custom dataset containing a mix of legitimate and phishing emails. Our results demonstrate that existing LLMs are capable of detecting and categorizing phishing emails, underscoring their potential in this domain.



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



## **10. Image Corruption-Inspired Membership Inference Attacks against Large Vision-Language Models**

cs.CV

Preprint. 15 pages

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.12340v2) [paper-pdf](http://arxiv.org/pdf/2506.12340v2)

**Authors**: Zongyu Wu, Minhua Lin, Zhiwei Zhang, Fali Wang, Xianren Zhang, Xiang Zhang, Suhang Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated outstanding performance in many downstream tasks. However, LVLMs are trained on large-scale datasets, which can pose privacy risks if training images contain sensitive information. Therefore, it is important to detect whether an image is used to train the LVLM. Recent studies have investigated membership inference attacks (MIAs) against LVLMs, including detecting image-text pairs and single-modality content. In this work, we focus on detecting whether a target image is used to train the target LVLM. We design simple yet effective Image Corruption-Inspired Membership Inference Attacks (ICIMIA) against LLVLMs, which are inspired by LVLM's different sensitivity to image corruption for member and non-member images. We first perform an MIA method under the white-box setting, where we can obtain the embeddings of the image through the vision part of the target LVLM. The attacks are based on the embedding similarity between the image and its corrupted version. We further explore a more practical scenario where we have no knowledge about target LVLMs and we can only query the target LVLMs with an image and a question. We then conduct the attack by utilizing the output text embeddings' similarity. Experiments on existing datasets validate the effectiveness of our proposed attack methods under those two different settings.



## **11. Mind the Inconspicuous: Revealing the Hidden Weakness in Aligned LLMs' Refusal Boundaries**

cs.AI

published at USENIX Security 25

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2405.20653v3) [paper-pdf](http://arxiv.org/pdf/2405.20653v3)

**Authors**: Jiahao Yu, Haozheng Luo, Jerry Yao-Chieh Hu, Wenbo Guo, Han Liu, Xinyu Xing

**Abstract**: Recent advances in Large Language Models (LLMs) have led to impressive alignment where models learn to distinguish harmful from harmless queries through supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). In this paper, we reveal a subtle yet impactful weakness in these aligned models. We find that simply appending multiple end of sequence (eos) tokens can cause a phenomenon we call context segmentation, which effectively shifts both harmful and benign inputs closer to the refusal boundary in the hidden space.   Building on this observation, we propose a straightforward method to BOOST jailbreak attacks by appending eos tokens. Our systematic evaluation shows that this strategy significantly increases the attack success rate across 8 representative jailbreak techniques and 16 open-source LLMs, ranging from 2B to 72B parameters. Moreover, we develop a novel probing mechanism for commercial APIs and discover that major providers such as OpenAI, Anthropic, and Qwen do not filter eos tokens, making them similarly vulnerable. These findings highlight a hidden yet critical blind spot in existing alignment and content filtering approaches.   We call for heightened attention to eos tokens' unintended influence on model behaviors, particularly in production systems. Our work not only calls for an input-filtering based defense, but also points to new defenses that make refusal boundaries more robust and generalizable, as well as fundamental alignment techniques that can defend against context segmentation attacks.



## **12. Distraction is All You Need for Multimodal Large Language Model Jailbreaking**

cs.CV

CVPR 2025 highlight

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2502.10794v2) [paper-pdf](http://arxiv.org/pdf/2502.10794v2)

**Authors**: Zuopeng Yang, Jiluan Fan, Anli Yan, Erdun Gao, Xin Lin, Tao Li, Kanghua Mo, Changyu Dong

**Abstract**: Multimodal Large Language Models (MLLMs) bridge the gap between visual and textual data, enabling a range of advanced applications. However, complex internal interactions among visual elements and their alignment with text can introduce vulnerabilities, which may be exploited to bypass safety mechanisms. To address this, we analyze the relationship between image content and task and find that the complexity of subimages, rather than their content, is key. Building on this insight, we propose the Distraction Hypothesis, followed by a novel framework called Contrasting Subimage Distraction Jailbreaking (CS-DJ), to achieve jailbreaking by disrupting MLLMs alignment through multi-level distraction strategies. CS-DJ consists of two components: structured distraction, achieved through query decomposition that induces a distributional shift by fragmenting harmful prompts into sub-queries, and visual-enhanced distraction, realized by constructing contrasting subimages to disrupt the interactions among visual elements within the model. This dual strategy disperses the model's attention, reducing its ability to detect and mitigate harmful content. Extensive experiments across five representative scenarios and four popular closed-source MLLMs, including GPT-4o-mini, GPT-4o, GPT-4V, and Gemini-1.5-Flash, demonstrate that CS-DJ achieves average success rates of 52.40% for the attack success rate and 74.10% for the ensemble attack success rate. These results reveal the potential of distraction-based approaches to exploit and bypass MLLMs' defenses, offering new insights for attack strategies.



## **13. Evaluating Large Language Models for Phishing Detection, Self-Consistency, Faithfulness, and Explainability**

cs.CR

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13746v1) [paper-pdf](http://arxiv.org/pdf/2506.13746v1)

**Authors**: Shova Kuikel, Aritran Piplai, Palvi Aggarwal

**Abstract**: Phishing attacks remain one of the most prevalent and persistent cybersecurity threat with attackers continuously evolving and intensifying tactics to evade the general detection system. Despite significant advances in artificial intelligence and machine learning, faithfully reproducing the interpretable reasoning with classification and explainability that underpin phishing judgments remains challenging. Due to recent advancement in Natural Language Processing, Large Language Models (LLMs) show a promising direction and potential for improving domain specific phishing classification tasks. However, enhancing the reliability and robustness of classification models requires not only accurate predictions from LLMs but also consistent and trustworthy explanations aligning with those predictions. Therefore, a key question remains: can LLMs not only classify phishing emails accurately but also generate explanations that are reliably aligned with their predictions and internally self-consistent? To answer these questions, we have fine-tuned transformer based models, including BERT, Llama models, and Wizard, to improve domain relevance and make them more tailored to phishing specific distinctions, using Binary Sequence Classification, Contrastive Learning (CL) and Direct Preference Optimization (DPO). To that end, we examined their performance in phishing classification and explainability by applying the ConsistenCy measure based on SHAPley values (CC SHAP), which measures prediction explanation token alignment to test the model's internal faithfulness and consistency and uncover the rationale behind its predictions and reasoning. Overall, our findings show that Llama models exhibit stronger prediction explanation token alignment with higher CC SHAP scores despite lacking reliable decision making accuracy, whereas Wizard achieves better prediction accuracy but lower CC SHAP scores.



## **14. Weakest Link in the Chain: Security Vulnerabilities in Advanced Reasoning Models**

cs.AI

Accepted to LLMSEC 2025

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13726v1) [paper-pdf](http://arxiv.org/pdf/2506.13726v1)

**Authors**: Arjun Krishna, Aaditya Rastogi, Erick Galinkin

**Abstract**: The introduction of advanced reasoning capabilities have improved the problem-solving performance of large language models, particularly on math and coding benchmarks. However, it remains unclear whether these reasoning models are more or less vulnerable to adversarial prompt attacks than their non-reasoning counterparts. In this work, we present a systematic evaluation of weaknesses in advanced reasoning models compared to similar non-reasoning models across a diverse set of prompt-based attack categories. Using experimental data, we find that on average the reasoning-augmented models are \emph{slightly more robust} than non-reasoning models (42.51\% vs 45.53\% attack success rate, lower is better). However, this overall trend masks significant category-specific differences: for certain attack types the reasoning models are substantially \emph{more vulnerable} (e.g., up to 32 percentage points worse on a tree-of-attacks prompt), while for others they are markedly \emph{more robust} (e.g., 29.8 points better on cross-site scripting injection). Our findings highlight the nuanced security implications of advanced reasoning in language models and emphasize the importance of stress-testing safety across diverse adversarial techniques.



## **15. On the Feasibility of Fully AI-automated Vishing Attacks**

cs.CR

To appear in AsiaCCS 2025

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2409.13793v2) [paper-pdf](http://arxiv.org/pdf/2409.13793v2)

**Authors**: João Figueiredo, Afonso Carvalho, Daniel Castro, Daniel Gonçalves, Nuno Santos

**Abstract**: A vishing attack is a form of social engineering where attackers use phone calls to deceive individuals into disclosing sensitive information, such as personal data, financial information, or security credentials. Attackers exploit the perceived urgency and authenticity of voice communication to manipulate victims, often posing as legitimate entities like banks or tech support. Vishing is a particularly serious threat as it bypasses security controls designed to protect information. In this work, we study the potential for vishing attacks to escalate with the advent of AI. In theory, AI-powered software bots may have the ability to automate these attacks by initiating conversations with potential victims via phone calls and deceiving them into disclosing sensitive information. To validate this thesis, we introduce ViKing, an AI-powered vishing system developed using publicly available AI technology. It relies on a Large Language Model (LLM) as its core cognitive processor to steer conversations with victims, complemented by a pipeline of speech-to-text and text-to-speech modules that facilitate audio-text conversion in phone calls. Through a controlled social experiment involving 240 participants, we discovered that ViKing has successfully persuaded many participants to reveal sensitive information, even those who had been explicitly warned about the risk of vishing campaigns. Interactions with ViKing's bots were generally considered realistic. From these findings, we conclude that tools like ViKing may already be accessible to potential malicious actors, while also serving as an invaluable resource for cyber awareness programs.



## **16. Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design**

cs.CR

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2504.10112v2) [paper-pdf](http://arxiv.org/pdf/2504.10112v2)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. Due to the opaque nature of LLMs, empirical methods are typically used to analyze their efficacy. The quality of this analysis is highly dependent on the chosen testbed, captured metrics and analysis methods employed.   This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 19 research papers detailing 18 prototypes and their respective testbeds.   We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios.



## **17. From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs**

cs.CR

10 pages

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13434v1) [paper-pdf](http://arxiv.org/pdf/2506.13434v1)

**Authors**: Alsharif Abuadbba, Chris Hicks, Kristen Moore, Vasilios Mavroudis, Burak Hasircioglu, Diksha Goel, Piers Jennings

**Abstract**: Large Language Models (LLMs) are set to reshape cybersecurity by augmenting red and blue team operations. Red teams can exploit LLMs to plan attacks, craft phishing content, simulate adversaries, and generate exploit code. Conversely, blue teams may deploy them for threat intelligence synthesis, root cause analysis, and streamlined documentation. This dual capability introduces both transformative potential and serious risks.   This position paper maps LLM applications across cybersecurity frameworks such as MITRE ATT&CK and the NIST Cybersecurity Framework (CSF), offering a structured view of their current utility and limitations. While LLMs demonstrate fluency and versatility across various tasks, they remain fragile in high-stakes, context-heavy environments. Key limitations include hallucinations, limited context retention, poor reasoning, and sensitivity to prompts, which undermine their reliability in operational settings.   Moreover, real-world integration raises concerns around dual-use risks, adversarial misuse, and diminished human oversight. Malicious actors could exploit LLMs to automate reconnaissance, obscure attack vectors, and lower the technical threshold for executing sophisticated attacks.   To ensure safer adoption, we recommend maintaining human-in-the-loop oversight, enhancing model explainability, integrating privacy-preserving mechanisms, and building systems robust to adversarial exploitation. As organizations increasingly adopt AI driven cybersecurity, a nuanced understanding of LLMs' risks and operational impacts is critical to securing their defensive value while mitigating unintended consequences.



## **18. Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs**

cs.CL

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13285v1) [paper-pdf](http://arxiv.org/pdf/2506.13285v1)

**Authors**: Houcheng Jiang, Zetong Zhao, Junfeng Fang, Haokai Ma, Ruipeng Wang, Yang Deng, Xiang Wang, Xiangnan He

**Abstract**: Large language models (LLMs) have shown strong performance across natural language tasks, but remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying parameters to map specific triggers to attacker-desired responses. However, these methods often suffer from safety fallback, where the model initially responds affirmatively but later reverts to refusals due to safety alignment. In this work, we propose DualEdit, a dual-objective model editing framework that jointly promotes affirmative outputs and suppresses refusal responses. To address two key challenges -- balancing the trade-off between affirmative promotion and refusal suppression, and handling the diversity of refusal expressions -- DualEdit introduces two complementary techniques. (1) Dynamic loss weighting calibrates the objective scale based on the pre-edited model to stabilize optimization. (2) Refusal value anchoring compresses the suppression target space by clustering representative refusal value vectors, reducing optimization conflict from overly diverse token sets. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and reduces safety fallback rate by 10.88\% over baselines.



## **19. Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks**

cs.AI

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13276v1) [paper-pdf](http://arxiv.org/pdf/2506.13276v1)

**Authors**: Yuefei Lyu, Chaozhuo Li, Xi Zhang, Tianle Zhang

**Abstract**: Text-attributed graphs (TAGs) integrate textual data with graph structures, providing valuable insights in applications such as social network analysis and recommendation systems. Graph Neural Networks (GNNs) effectively capture both topological structure and textual information in TAGs but are vulnerable to adversarial attacks. Existing graph injection attack (GIA) methods assume that attackers can directly manipulate the embedding layer, producing non-explainable node embeddings. Furthermore, the effectiveness of these attacks often relies on surrogate models with high training costs. Thus, this paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs. Our approach leverages large language models (LLMs) to generate interpretable text-level node attributes directly, ensuring attacks remain feasible in real-world scenarios. We design strategies for LLM prompting that balance exploration and reliability to guide text generation, and propose a similarity assessment method to evaluate attack text effectiveness in disrupting graph homophily. This method efficiently perturbs the target node with minimal training costs in a strict black-box setting, ensuring a text-level graph injection attack for TAGs. Experiments on real-world TAG datasets validate the superior performance of ATAG-LLM compared to state-of-the-art embedding-level and text-level attack methods.



## **20. Detecting Hard-Coded Credentials in Software Repositories via LLMs**

cs.CR

Accepted to the ACM Digital Threats: Research and Practice (DTRAP)

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13090v1) [paper-pdf](http://arxiv.org/pdf/2506.13090v1)

**Authors**: Chidera Biringa, Gokhan Kul

**Abstract**: Software developers frequently hard-code credentials such as passwords, generic secrets, private keys, and generic tokens in software repositories, even though it is strictly advised against due to the severe threat to the security of the software. These credentials create attack surfaces exploitable by a potential adversary to conduct malicious exploits such as backdoor attacks. Recent detection efforts utilize embedding models to vectorize textual credentials before passing them to classifiers for predictions. However, these models struggle to discriminate between credentials with contextual and complex sequences resulting in high false positive predictions. Context-dependent Pre-trained Language Models (PLMs) or Large Language Models (LLMs) such as Generative Pre-trained Transformers (GPT) tackled this drawback by leveraging the transformer neural architecture capacity for self-attention to capture contextual dependencies between words in input sequences. As a result, GPT has achieved wide success in several natural language understanding endeavors. Hence, we assess LLMs to represent these observations and feed extracted embedding vectors to a deep learning classifier to detect hard-coded credentials. Our model outperforms the current state-of-the-art by 13% in F1 measure on the benchmark dataset. We have made all source code and data publicly available to facilitate the reproduction of all results presented in this paper.



## **21. Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions**

cs.LG

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2502.04322v2) [paper-pdf](http://arxiv.org/pdf/2502.04322v2)

**Authors**: Yik Siu Chan, Narutatsu Ri, Yuxin Xiao, Marzyeh Ghassemi

**Abstract**: Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions.



## **22. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12880v1) [paper-pdf](http://arxiv.org/pdf/2506.12880v1)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack (Zou et al., 2023), we observe that suffixes vary in efficacy: some markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that GCG's effectiveness is driven by a shallow, critical mechanism, built on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving attack success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.



## **23. Transforming Chatbot Text: A Sequence-to-Sequence Approach**

cs.CL

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12843v1) [paper-pdf](http://arxiv.org/pdf/2506.12843v1)

**Authors**: Natesh Reddy, Mark Stamp

**Abstract**: Due to advances in Large Language Models (LLMs) such as ChatGPT, the boundary between human-written text and AI-generated text has become blurred. Nevertheless, recent work has demonstrated that it is possible to reliably detect GPT-generated text. In this paper, we adopt a novel strategy to adversarially transform GPT-generated text using sequence-to-sequence (Seq2Seq) models, with the goal of making the text more human-like. We experiment with the Seq2Seq models T5-small and BART which serve to modify GPT-generated sentences to include linguistic, structural, and semantic components that may be more typical of human-authored text. Experiments show that classification models trained to distinguish GPT-generated text are significantly less accurate when tested on text that has been modified by these Seq2Seq models. However, after retraining classification models on data generated by our Seq2Seq technique, the models are able to distinguish the transformed GPT-generated text from human-generated text with high accuracy. This work adds to the accumulating knowledge of text transformation as a tool for both attack -- in the sense of defeating classification models -- and defense -- in the sense of improved classifiers -- thereby advancing our understanding of AI-generated text.



## **24. I Know What You Said: Unveiling Hardware Cache Side-Channels in Local Large Language Model Inference**

cs.CR

Submitted for review in January 22, 2025, revised under shepherding

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2505.06738v3) [paper-pdf](http://arxiv.org/pdf/2505.06738v3)

**Authors**: Zibo Gao, Junjie Hu, Feng Guo, Yixin Zhang, Yinglong Han, Siyuan Liu, Haiyang Li, Zhiqiang Lv

**Abstract**: Large Language Models (LLMs) that can be deployed locally have recently gained popularity for privacy-sensitive tasks, with companies such as Meta, Google, and Intel playing significant roles in their development. However, the security of local LLMs through the lens of hardware cache side-channels remains unexplored. In this paper, we unveil novel side-channel vulnerabilities in local LLM inference: token value and token position leakage, which can expose both the victim's input and output text, thereby compromising user privacy. Specifically, we found that adversaries can infer the token values from the cache access patterns of the token embedding operation, and deduce the token positions from the timing of autoregressive decoding phases. To demonstrate the potential of these leaks, we design a novel eavesdropping attack framework targeting both open-source and proprietary LLM inference systems. The attack framework does not directly interact with the victim's LLM and can be executed without privilege.   We evaluate the attack on a range of practical local LLM deployments (e.g., Llama, Falcon, and Gemma), and the results show that our attack achieves promising accuracy. The restored output and input text have an average edit distance of 5.2% and 17.3% to the ground truth, respectively. Furthermore, the reconstructed texts achieve average cosine similarity scores of 98.7% (input) and 98.0% (output).



## **25. Can We Infer Confidential Properties of Training Data from LLMs?**

cs.LG

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.10364v2) [paper-pdf](http://arxiv.org/pdf/2506.10364v2)

**Authors**: Pengrun Huang, Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri

**Abstract**: Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.



## **26. SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression**

cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12707v1) [paper-pdf](http://arxiv.org/pdf/2506.12707v1)

**Authors**: Yucheng Li, Surin Ahn, Huiqiang Jiang, Amir H. Abdi, Yuqing Yang, Lili Qiu

**Abstract**: Large language models (LLMs) have achieved widespread adoption across numerous applications. However, many LLMs are vulnerable to malicious attacks even after safety alignment. These attacks typically bypass LLMs' safety guardrails by wrapping the original malicious instructions inside adversarial jailbreaks prompts. Previous research has proposed methods such as adversarial training and prompt rephrasing to mitigate these safety vulnerabilities, but these methods often reduce the utility of LLMs or lead to significant computational overhead and online latency. In this paper, we propose SecurityLingua, an effective and efficient approach to defend LLMs against jailbreak attacks via security-oriented prompt compression. Specifically, we train a prompt compressor designed to discern the "true intention" of the input prompt, with a particular focus on detecting the malicious intentions of adversarial prompts. Then, in addition to the original prompt, the intention is passed via the system prompt to the target LLM to help it identify the true intention of the request. SecurityLingua ensures a consistent user experience by leaving the original input prompt intact while revealing the user's potentially malicious intention and stimulating the built-in safety guardrails of the LLM. Moreover, thanks to prompt compression, SecurityLingua incurs only a negligible overhead and extra token cost compared to all existing defense methods, making it an especially practical solution for LLM defense. Experimental results demonstrate that SecurityLingua can effectively defend against malicious attacks and maintain utility of the LLM with negligible compute and latency overhead. Our code is available at https://aka.ms/SecurityLingua.



## **27. Alphabet Index Mapping: Jailbreaking LLMs through Semantic Dissimilarity**

cs.CR

10 pages, 2 figures, 3 tables

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12685v1) [paper-pdf](http://arxiv.org/pdf/2506.12685v1)

**Authors**: Bilal Saleh Husain

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their susceptibility to adversarial attacks, particularly jailbreaking, poses significant safety and ethical concerns. While numerous jailbreak methods exist, many suffer from computational expense, high token usage, or complex decoding schemes. Liu et al. (2024) introduced FlipAttack, a black-box method that achieves high attack success rates (ASR) through simple prompt manipulation. This paper investigates the underlying mechanisms of FlipAttack's effectiveness by analyzing the semantic changes induced by its flipping modes. We hypothesize that semantic dissimilarity between original and manipulated prompts is inversely correlated with ASR. To test this, we examine embedding space visualizations (UMAP, KDE) and cosine similarities for FlipAttack's modes. Furthermore, we introduce a novel adversarial attack, Alphabet Index Mapping (AIM), designed to maximize semantic dissimilarity while maintaining simple decodability. Experiments on GPT-4 using a subset of AdvBench show AIM and its variant AIM+FWO achieve a 94% ASR, outperforming FlipAttack and other methods on this subset. Our findings suggest that while high semantic dissimilarity is crucial, a balance with decoding simplicity is key for successful jailbreaking. This work contributes to a deeper understanding of adversarial prompt mechanics and offers a new, effective jailbreak technique.



## **28. MEraser: An Effective Fingerprint Erasure Approach for Large Language Models**

cs.CR

Accepted by ACL 2025, Main Conference, Long Paper

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12551v1) [paper-pdf](http://arxiv.org/pdf/2506.12551v1)

**Authors**: Jingxuan Zhang, Zhenhua Xu, Rui Hu, Wenpeng Xing, Xuhong Zhang, Meng Han

**Abstract**: Large Language Models (LLMs) have become increasingly prevalent across various sectors, raising critical concerns about model ownership and intellectual property protection. Although backdoor-based fingerprinting has emerged as a promising solution for model authentication, effective attacks for removing these fingerprints remain largely unexplored. Therefore, we present Mismatched Eraser (MEraser), a novel method for effectively removing backdoor-based fingerprints from LLMs while maintaining model performance. Our approach leverages a two-phase fine-tuning strategy utilizing carefully constructed mismatched and clean datasets. Through extensive evaluation across multiple LLM architectures and fingerprinting methods, we demonstrate that MEraser achieves complete fingerprinting removal while maintaining model performance with minimal training data of fewer than 1,000 samples. Furthermore, we introduce a transferable erasure mechanism that enables effective fingerprinting removal across different models without repeated training. In conclusion, our approach provides a practical solution for fingerprinting removal in LLMs, reveals critical vulnerabilities in current fingerprinting techniques, and establishes comprehensive evaluation benchmarks for developing more resilient model protection methods in the future.



## **29. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

cs.CR

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12430v1) [paper-pdf](http://arxiv.org/pdf/2506.12430v1)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.



## **30. Exploring the Secondary Risks of Large Language Models**

cs.LG

18 pages, 5 figures

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12382v1) [paper-pdf](http://arxiv.org/pdf/2506.12382v1)

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.



## **31. Stepwise Reasoning Error Disruption Attack of LLMs**

cs.AI

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2412.11934v5) [paper-pdf](http://arxiv.org/pdf/2412.11934v5)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications. Our code is available at: https://github.com/Applied-Machine-Learning-Lab/SEED-Attack.



## **32. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2411.18688v5) [paper-pdf](http://arxiv.org/pdf/2411.18688v5)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Alvaro Velasquez, Ahmad Beirami, Furong Huang, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.



## **33. AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents**

cs.CR

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2505.05849v4) [paper-pdf](http://arxiv.org/pdf/2505.05849v4)

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentVigil, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentVigil on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentVigil exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.



## **34. QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety**

cs.CR

Accept to ACLW 2025 (WOAH)

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12299v1) [paper-pdf](http://arxiv.org/pdf/2506.12299v1)

**Authors**: Taegyeong Lee, Jeonghwa Yoo, Hyoungseo Cho, Soo Yong Kim, Yunho Maeng

**Abstract**: The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.



## **35. InfoFlood: Jailbreaking Large Language Models with Information Overload**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.12274v1) [paper-pdf](http://arxiv.org/pdf/2506.12274v1)

**Authors**: Advait Yadav, Haibo Jin, Man Luo, Jun Zhuang, Haohan Wang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. However, their potential to generate harmful responses has raised significant societal and regulatory concerns, especially when manipulated by adversarial techniques known as "jailbreak" attacks. Existing jailbreak methods typically involve appending carefully crafted prefixes or suffixes to malicious prompts in order to bypass the built-in safety mechanisms of these models.   In this work, we identify a new vulnerability in which excessive linguistic complexity can disrupt built-in safety mechanisms-without the need for any added prefixes or suffixes-allowing attackers to elicit harmful outputs directly. We refer to this phenomenon as Information Overload.   To automatically exploit this vulnerability, we propose InfoFlood, a jailbreak attack that transforms malicious queries into complex, information-overloaded queries capable of bypassing built-in safety mechanisms. Specifically, InfoFlood: (1) uses linguistic transformations to rephrase malicious queries, (2) identifies the root cause of failure when an attempt is unsuccessful, and (3) refines the prompt's linguistic structure to address the failure while preserving its malicious intent.   We empirically validate the effectiveness of InfoFlood on four widely used LLMs-GPT-4o, GPT-3.5-turbo, Gemini 2.0, and LLaMA 3.1-by measuring their jailbreak success rates. InfoFlood consistently outperforms baseline attacks, achieving up to 3 times higher success rates across multiple jailbreak benchmarks. Furthermore, we demonstrate that commonly adopted post-processing defenses, including OpenAI's Moderation API, Perspective API, and SmoothLLM, fail to mitigate these attacks. This highlights a critical weakness in traditional AI safety guardrails when confronted with information overload-based jailbreaks.



## **36. Improving Large Language Model Safety with Contrastive Representation Learning**

cs.CL

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11938v1) [paper-pdf](http://arxiv.org/pdf/2506.11938v1)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense



## **37. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2501.18638v2) [paper-pdf](http://arxiv.org/pdf/2501.18638v2)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.



## **38. Black-Box Adversarial Attacks on LLM-Based Code Completion**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2408.02509v2) [paper-pdf](http://arxiv.org/pdf/2408.02509v2)

**Authors**: Slobodan Jenko, Niels Mündler, Jingxuan He, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.



## **39. TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks**

cs.LG

12 pages, 5 figures, in KDD 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11844v1) [paper-pdf](http://arxiv.org/pdf/2506.11844v1)

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field.



## **40. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

cs.CL

Accepted to ACL 2025 Findings

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2406.14023v4) [paper-pdf](http://arxiv.org/pdf/2406.14023v4)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.



## **41. Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11521v1) [paper-pdf](http://arxiv.org/pdf/2506.11521v1)

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense.



## **42. DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents**

cs.CR

18 pages, 12 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.12104v1) [paper-pdf](http://arxiv.org/pdf/2506.12104v1)

**Authors**: Hao Li, Xiaogeng Liu, Hung-Chun Chiu, Dianqi Li, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) are increasingly central to agentic systems due to their strong reasoning and planning capabilities. By interacting with external environments through predefined tools, these agents can carry out complex user tasks. Nonetheless, this interaction also introduces the risk of prompt injection attacks, where malicious inputs from external sources can mislead the agent's behavior, potentially resulting in economic loss, privacy leakage, or system compromise. System-level defenses have recently shown promise by enforcing static or predefined policies, but they still face two key challenges: the ability to dynamically update security rules and the need for memory stream isolation. To address these challenges, we propose DRIFT, a Dynamic Rule-based Isolation Framework for Trustworthy agentic systems, which enforces both control- and data-level constraints. A Secure Planner first constructs a minimal function trajectory and a JSON-schema-style parameter checklist for each function node based on the user query. A Dynamic Validator then monitors deviations from the original plan, assessing whether changes comply with privilege limitations and the user's intent. Finally, an Injection Isolator detects and masks any instructions that may conflict with the user query from the memory stream to mitigate long-term risks. We empirically validate the effectiveness of DRIFT on the AgentDojo benchmark, demonstrating its strong security performance while maintaining high utility across diverse models -- showcasing both its robustness and adaptability.



## **43. Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs**

cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11415v1) [paper-pdf](http://arxiv.org/pdf/2506.11415v1)

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou

**Abstract**: In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system.



## **44. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2405.06823v3) [paper-pdf](http://arxiv.org/pdf/2405.06823v3)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.



## **45. Improving LLM Safety Alignment with Dual-Objective Optimization**

cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2503.03710v2) [paper-pdf](http://arxiv.org/pdf/2503.03710v2)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment



## **46. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2401.17256v3) [paper-pdf](http://arxiv.org/pdf/2401.17256v3)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **47. PRSA: Prompt Stealing Attacks against Real-World Prompt Services**

cs.CR

This is the extended version of the paper accepted at the 34th USENIX  Security Symposium (USENIX Security 2025)

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2402.19200v3) [paper-pdf](http://arxiv.org/pdf/2402.19200v3)

**Authors**: Yong Yang, Changjiang Li, Qingming Li, Oubo Ma, Haoyu Wang, Zonghui Wang, Yandong Gao, Wenzhi Chen, Shouling Ji

**Abstract**: Recently, large language models (LLMs) have garnered widespread attention for their exceptional capabilities. Prompts are central to the functionality and performance of LLMs, making them highly valuable assets. The increasing reliance on high-quality prompts has driven significant growth in prompt services. However, this growth also expands the potential for prompt leakage, increasing the risk that attackers could replicate original functionalities, create competing products, and severely infringe on developers' intellectual property. Despite these risks, prompt leakage in real-world prompt services remains underexplored.   In this paper, we present PRSA, a practical attack framework designed for prompt stealing. PRSA infers the detailed intent of prompts through very limited input-output analysis and can successfully generate stolen prompts that replicate the original functionality. Extensive evaluations demonstrate PRSA's effectiveness across two main types of real-world prompt services. Specifically, compared to previous works, it improves the attack success rate from 17.8% to 46.1% in prompt marketplaces and from 39% to 52% in LLM application stores, respectively. Notably, in the attack on "Math", one of the most popular educational applications in OpenAI's GPT Store with over 1 million conversations, PRSA uncovered a hidden Easter egg that had not been revealed previously. Besides, our analysis reveals that higher mutual information between a prompt and its output correlates with an increased risk of leakage. This insight guides the design and evaluation of two potential defenses against the security threats posed by PRSA. We have reported these findings to the prompt service vendors, including PromptBase and OpenAI, and actively collaborate with them to implement defensive measures.



## **48. Unsourced Adversarial CAPTCHA: A Bi-Phase Adversarial CAPTCHA Framework**

cs.CV

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10685v1) [paper-pdf](http://arxiv.org/pdf/2506.10685v1)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: With the rapid advancements in deep learning, traditional CAPTCHA schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on original image characteristics, resulting in distortions that hinder human interpretation and limit applicability in scenarios lacking initial input images. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (UAC), a novel framework generating high-fidelity adversarial examples guided by attacker-specified text prompts. Leveraging a Large Language Model (LLM), UAC enhances CAPTCHA diversity and supports both targeted and untargeted attacks. For targeted attacks, the EDICT method optimizes dual latent variables in a diffusion model for superior image quality. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-UAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show BP-UAC achieves high attack success rates across diverse systems, generating natural CAPTCHAs indistinguishable to humans and DNNs.



## **49. SoK: Evaluating Jailbreak Guardrails for Large Language Models**

cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10597v1) [paper-pdf](http://arxiv.org/pdf/2506.10597v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Wenxuan Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety mechanisms. Guardrails--external defense mechanisms that monitor and control LLM interaction--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, explore their universality across attack types, and provide insights into optimizing defense combinations. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at https://github.com/xunguangwang/SoK4JailbreakGuardrails.



## **50. Towards Action Hijacking of Large Language Model-based Agent**

cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2412.10807v2) [paper-pdf](http://arxiv.org/pdf/2412.10807v2)

**Authors**: Yuyang Zhang, Kangjie Chen, Jiaxin Gao, Ronghao Cui, Run Wang, Lina Wang, Tianwei Zhang

**Abstract**: Recently, applications powered by Large Language Models (LLMs) have made significant strides in tackling complex tasks. By harnessing the advanced reasoning capabilities and extensive knowledge embedded in LLMs, these applications can generate detailed action plans that are subsequently executed by external tools. Furthermore, the integration of retrieval-augmented generation (RAG) enhances performance by incorporating up-to-date, domain-specific knowledge into the planning and execution processes. This approach has seen widespread adoption across various sectors, including healthcare, finance, and software development. Meanwhile, there are also growing concerns regarding the security of LLM-based applications. Researchers have disclosed various attacks, represented by jailbreak and prompt injection, to hijack the output actions of these applications. Existing attacks mainly focus on crafting semantically harmful prompts, and their validity could diminish when security filters are employed. In this paper, we introduce AI$\mathbf{^2}$, a novel attack to manipulate the action plans of LLM-based applications. Different from existing solutions, the innovation of AI$\mathbf{^2}$ lies in leveraging the knowledge from the application's database to facilitate the construction of malicious but semantically-harmless prompts. To this end, it first collects action-aware knowledge from the victim application. Based on such knowledge, the attacker can generate misleading input, which can mislead the LLM to generate harmful action plans, while bypassing possible detection mechanisms easily. Our evaluations on three real-world applications demonstrate the effectiveness of AI$\mathbf{^2}$: it achieves an average attack success rate of 84.30\% with the best of 99.70\%. Besides, it gets an average bypass rate of 92.7\% against common safety filters and 59.45\% against dedicated defense.



