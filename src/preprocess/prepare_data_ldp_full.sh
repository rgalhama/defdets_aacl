echo "Preparing LDP data..."

#Original (raw) data
rawdata=`realpath ~/Data_Research/LDP/all_ldp_det_n_data_3_2_2022.csv`
sep=';'
subjectcol=2
parentcol=4
childrencol=5
sessioncol=10
detchildrencol=8
detparentcol=9

#Processed data
datafolder='../../data/processed/LDP/'
refdata=$datafolder"/all_ldp_det_n_data_3_2_2022_changedmasks.csv"

#Change MASK with PIIMASK to avoid confusion with language modelling masks
cat $rawdata | sed 's/MASK/PIIMASK/g' > $refdata

# Extract caregiver's utterances (used to train tokenizer)
cat $refdata | cut -d${sep} -f${parentcol} | sed 1d | sed -r '/^\s*$/d' > ${datafolder}'ldp_parents.txt'


# Extract parents utterances / session
mkdir $datafolder'/sessions'
for session in $(seq 1 12);
do
  cat $refdata | cut -d${sep} -f${parentcol},${sessioncol} | grep "${sep}${session}$"| sed '/^;/d;s/;.*$//g' > parent_utterances_session_${session}.csv
  mv parent_utterances_session_*.csv $datafolder'/sessions'
done

# Extract children's sentences (to test model)
cat $refdata | cut -d${sep} -f${subjectcol},${sessioncol},${childrencol},${detchildrencol} | sed '/;;;/d' > children_utterances_session.csv
mv children_utterances_session.csv $datafolder

# Extract parents's sentences in the same format as above (to test model on training data)
cat $refdata | cut -d${sep} -f${subjectcol},${sessioncol},${parentcol},${detparentcol} | sed '/;;;/d' > parents_utterances_session.csv
mv parents_utterances_session.csv $datafolder

# Extract only det-noun combinations (to compute overlap)
#Only for det_noun combinations: extract session, parent_utt, det_noun
cat $refdata | cut -d${sep} -f${subjectcol},${parentcol},${detparentcol},${sessioncol} | grep -v -e ';;' | sed 's/;/,/g' > parents_detnoun_session.csv
mv parents_detnoun_session.csv $datafolder

#Only for det_noun combinations: extract session, children_utt, det_noun
cat $refdata | cut -d${sep} -f${subjectcol},${childrencol},${detchildrencol},${sessioncol} | grep -v -e ';;' | sed 's/;/,/g' > children_detnoun_session.csv
mv children_detnoun_session.csv $datafolder

