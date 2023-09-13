#Paths (adjust if needed!)
curr_dir=`pwd`
raw_datafolder=~/Data_Research/Manchester/
project_datafolder="../../data/"
processed_datafolder="../../data/processed/Manchester/"
if [ ! -d "$processed_datafolder" ]; then
  mkdir $processed_datafolder
fi
#Get full paths:
cd $processed_datafolder;processed_datafolder=`pwd`;cd $curr_dir
cd $project_datafolder;project_datafolder=`pwd`;cd $curr_dir


## For non-incremental analysis ##
##################################

extract_manchester_nonincremental () {
  ##Extract parents sentences (MOT)
  cd $raw_datafolder
  for kidfolder in `ls -d */`; do
     kid=`echo $kidfolder | sed 's/\///g'`
     cat "${kidfolder}${kid}.cha" | grep ^\*MOT | sed 's/^\*MOT:[ \t]*//g' > "${kid}_MOT.cha"
  done

  #Move to processed data folder
  mv *_MOT.cha ${processed_datafolder}

  #Prepare full corpus of parents sentences
  cd $processed_datafolder
  if [ ! -d "MOT_ALL" ]; then
    mkdir MOT_ALL
  fi
  cat *_MOT.cha > MOT_ALL/all_MOT.cha

}

## For incremental analysis ##
##############################
group_files_in_phases () {
  cd $raw_datafolder


  for kidfolder in `ls -d */`; do
    kid=`echo $kidfolder | sed 's/\///g'`
    cd $kidfolder

    #create folder structure for phase-divided data
    for n in {1..6}; do
      if [ -d $n ]; then
        rm -r $n
      fi
        mkdir $n
    done

    for fname in `ls -f 0*.cha`; do
     is_dummy=`cat $fname | grep -c 'This is a dummy file'`
     if [[ "$is_dummy" != "1" ]]
     then
       #1. find age
       age_long=`cat $fname | grep ^\@ID | grep CHI  | cut -d'|' -f4`
       years=`echo $age_long | cut -f1 -d';' | sed 's/ *$//g'`
       months=`echo $age_long | cut -f2 -d';' | cut -f1 -d'.' | sed 's/ *$//g'`
       age_in_months=`echo -e $years*12+$months | bc`
       #echo $age_in_months

       #2. find phase
#        phase_found=0
#        for n in {1..6}; do
          lines="`cat $processed_datafolder"/phases.csv" | grep -i $kid`"
          for line in $lines; do
            upper=`echo $line| cut -f4 -d','`
            if [[ $age_in_months -le $upper ]]
            then
            phase=`echo $line| cut -f3 -d','`
            phase_found=1
            break
            fi
          done
#          if [[ $phase_found -eq 1 ]]
#          then
#            break
#          fi
#        done
        #3. move to corresponding phase folder
        cp $fname $phase
        #4. cat parents productions into corresponding phase file
        if [ ! -d "${processed_datafolder}/by_phase/" ]; then
          mkdir "${processed_datafolder}/by_phase/"
        fi
        cat $fname | grep ^\*MOT | sed 's/^\*MOT:[ \t]*//g' >> "${processed_datafolder}/by_phase/${phase}_MOT.cha"

     fi
     done
    cd $raw_datafolder
  done
}

extract_manchester_incremental () {

  #Put each file in phase folder, according to criteria in phases.csv
  group_files_in_phases


  #Extract parental data for each phase

  #Extract children data for each phase


}
## main ##
##########

#extract_manchester_nonincremental
extract_manchester_incremental

#Adjust columns in analyzed data
cd $project_datafolder
cat manchester_child_lemma_nps_data_NO_PLURALS.csv | sed 's/manchester\_child/subject/' > aux
mv aux $processed_datafolder/manchester_child_lemma_nps_data_NO_PLURALS.csv

cat manchester_mother_lemma_nps_data_NO_PLURALS.csv | sed 's/manchester\_child/subject/' > aux
mv aux $processed_datafolder/manchester_mother_lemma_nps_data_NO_PLURALS.csv
