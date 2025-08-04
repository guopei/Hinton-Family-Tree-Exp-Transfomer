import torch
import random

relationships = [
    ("Penelope", "has_husband", ["Christopher"]),
    ("Christine", "has_husband", ["Andrew"]),
    ("Margaret", "has_husband", ["Arthur"]),
    ("Victoria", "has_husband", ["James"]),
    ("Jennifer", "has_husband", ["Charles"]),
    ("Maria", "has_husband", ["Roberto"]),
    ("Francesca", "has_husband", ["Pierro"]),
    ("Gina", "has_husband", ["Emilio"]),
    ("Lucia", "has_husband", ["Marco"]),
    ("Angela", "has_husband", ["Tomaso"]),
    ("Christopher", "has_wife", ["Penelope"]),
    ("Andrew", "has_wife", ["Christine"]),
    ("Arthur", "has_wife", ["Margaret"]),
    ("James", "has_wife", ["Victoria"]),
    ("Charles", "has_wife", ["Jennifer"]),
    ("Roberto", "has_wife", ["Maria"]),
    ("Pierro", "has_wife", ["Francesca"]),
    ("Emilio", "has_wife", ["Gina"]),
    ("Marco", "has_wife", ["Lucia"]),
    ("Tomaso", "has_wife", ["Angela"]),
    ("Christopher", "has_son", ["Arthur"]),
    ("Penelope", "has_son", ["Arthur"]),
    ("Andrew", "has_son", ["James"]),
    ("Christine", "has_son", ["James"]),
    ("Victoria", "has_son", ["Colin"]),
    ("James", "has_son", ["Colin"]),
    ("Roberto", "has_son", ["Emilio"]),
    ("Maria", "has_son", ["Emilio"]),
    ("Lucia", "has_son", ["Alfonso"]),
    ("Marco", "has_son", ["Alfonso"]),
    ("Pierro", "has_son", ["Tomaso"]),
    ("Francesca", "has_son", ["Tomaso"]),
    ("Christopher", "has_daughter", ["Victoria"]),
    ("Penelope", "has_daughter", ["Victoria"]),
    ("Andrew", "has_daughter", ["Jennifer"]),
    ("Christine", "has_daughter", ["Jennifer"]),
    ("Victoria", "has_daughter", ["Charlotte"]),
    ("James", "has_daughter", ["Charlotte"]),
    ("Roberto", "has_daughter", ["Lucia"]),
    ("Maria", "has_daughter", ["Lucia"]),
    ("Lucia", "has_daughter", ["Sophia"]),
    ("Marco", "has_daughter", ["Sophia"]),
    ("Pierro", "has_daughter", ["Angela"]),
    ("Francesca", "has_daughter", ["Angela"]),
    ("Arthur", "has_father", ["Christopher"]),
    ("Victoria", "has_father", ["Christopher"]),
    ("James", "has_father", ["Andrew"]),
    ("Jennifer", "has_father", ["Andrew"]),
    ("Colin", "has_father", ["James"]),
    ("Charlotte", "has_father", ["James"]),
    ("Emilio", "has_father", ["Roberto"]),
    ("Lucia", "has_father", ["Roberto"]),
    ("Alfonso", "has_father", ["Marco"]),
    ("Sophia", "has_father", ["Marco"]),
    ("Marco", "has_father", ["Pierro"]),
    ("Angela", "has_father", ["Pierro"]),
    ("Arthur", "has_mother", ["Penelope"]),
    ("Victoria", "has_mother", ["Penelope"]),
    ("James", "has_mother", ["Christine"]),
    ("Jennifer", "has_mother", ["Christine"]),
    ("Colin", "has_mother", ["Victoria"]),
    ("Charlotte", "has_mother", ["Victoria"]),
    ("Emilio", "has_mother", ["Maria"]),
    ("Lucia", "has_mother", ["Maria"]),
    ("Alfonso", "has_mother", ["Lucia"]),
    ("Sophia", "has_mother", ["Lucia"]),
    ("Marco", "has_mother", ["Francesca"]),
    ("Angela", "has_mother", ["Francesca"]),
    ("Colin", "has_uncle", ["Arthur", "Charles"]),
    ("Charlotte", "has_uncle", ["Arthur", "Charles"]),
    ("Alfonso", "has_uncle", ["Emilio", "Tomaso"]),
    ("Sophia", "has_uncle", ["Emilio", "Tomaso"]),
    ("Colin", "has_aunt", ["Jennifer", "Margaret"]),
    ("Charlotte", "has_aunt", ["Jennifer", "Margaret"]),
    ("Sophia", "has_aunt", ["Angela", "Gina"]),
    ("Alfonso", "has_aunt", ["Angela", "Gina"]),
    ("Jennifer", "has_niece", ["Charlotte"]),
    ("Arthur", "has_niece", ["Charlotte"]),
    ("Margaret", "has_niece", ["Charlotte"]),
    ("Charles", "has_niece", ["Charlotte"]),
    ("Angela", "has_niece", ["Sophia"]),
    ("Emilio", "has_niece", ["Sophia"]),
    ("Gina", "has_niece", ["Sophia"]),
    ("Tomaso", "has_niece", ["Sophia"]),
    ("Arthur", "has_nephew", ["Colin"]),
    ("Jennifer", "has_nephew", ["Colin"]),
    ("Margaret", "has_nephew", ["Colin"]),
    ("Charles", "has_nephew", ["Colin"]),
    ("Angela", "has_nephew", ["Alfonso"]),
    ("Emilio", "has_nephew", ["Alfonso"]),
    ("Gina", "has_nephew", ["Alfonso"]),
    ("Tomaso", "has_nephew", ["Alfonso"]),
    ("Victoria", "has_brother", ["Arthur"]),
    ("Jennifer", "has_brother", ["James"]),
    ("Charlotte", "has_brother", ["Colin"]),
    ("Lucia", "has_brother", ["Emilio"]),
    ("Sophia", "has_brother", ["Alfonso"]),
    ("Angela", "has_brother", ["Marco"]),
    ("Arthur", "has_sister", ["Victoria"]),
    ("James", "has_sister", ["Jennifer"]),
    ("Colin", "has_sister", ["Charlotte"]),
    ("Emilio", "has_sister", ["Lucia"]),
    ("Alfonso", "has_sister", ["Sophia"]),
    ("Marco", "has_sister", ["Angela"]),
]

names = sorted(list(set([name for name, _, _ in relationships])))
relations = sorted(list(set([relation for _, relation, _ in relationships])))
name_relations = names + relations

name_relations_to_index = {name_relation: i for i, name_relation in enumerate(name_relations)}

voc_size = len(name_relations)
block_size = 2
train_num = 100
dataset_size = len(relationships)

def prepare_data():
    random.shuffle(relationships)

    word_sequences = torch.zeros(dataset_size, (block_size+1), dtype=torch.long)

    for i, (name_input, relation_input, name_output) in enumerate(relationships):
        word_sequences[i][0] = name_relations_to_index[name_input]
        word_sequences[i][1] = name_relations_to_index[relation_input]
        word_sequences[i][2] = name_relations_to_index[name_output[0]]

    train_inputs = word_sequences[:train_num][:, :block_size]
    train_outputs = word_sequences[:train_num][:, -block_size:]

    test_inputs = word_sequences[train_num:][:, :block_size]
    test_outputs = word_sequences[train_num:][:, -block_size:]

    test_outputs_tensor = torch.zeros(test_outputs.shape[0], test_outputs.shape[1], voc_size, dtype=torch.long)
    for i in range(test_outputs.shape[0]):
        for j in range(test_outputs.shape[1]):
            test_outputs_tensor[i][j][test_outputs[i][j]] = 1

    return train_inputs, train_outputs, test_inputs, test_outputs_tensor