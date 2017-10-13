import sys
import rdflib
import pdb
import gzip

#print list(fb.query('select distinct (count(?o) as ?count) where{<http://rdf.freebase.com/ns/m.03m8s83> <http://rdf.freebase.com/ns/type.object.key> ?o}'))
def get_code_to_english(path):
	f=open(path,'r')
	fbmap_code_eng={}
	fbmap_eng_code={}
#	file_temp=open('./wrong.txt','w')
	for index,line in enumerate(f):
		value_list=line.split('\t')
		if len(value_list)==2:
			key=value_list[0].strip()
			value=value_list[1].strip()
		#	if fbmap_code_eng.has_key(key):
#				file_temp.write(line)
		#		continue
			fbmap_code_eng[key]=value
			fbmap_eng_code[value]=key
		else:
			print('line in data file error')
			pdb.set_trace()
#	file_temp.close()
	return fbmap_code_eng, fbmap_eng_code

def read_rdf_kb(eng_code,entities):
	'''
	search both subjects and objects from knowledge base by the entities
	'''
	PREFIX='''
		PREFIX : <http://rdf.freebase.com/ns/>
		PREFIX ns: <http://rdf.freebase.com/ns/>
	'''
	fb=rdflib.Graph('Sleepycat',identifier='freebase')
	fb.open('/data/analysis/qichuan_data/freebase/db')
	result={}
	result_sub={}
	result_obj={}
	for entity in entities:
	#	print(entity)
		if eng_code.has_key(entity):
			key=eng_code[entity]
			key=key.replace('/','.')
		#	pdb.set_trace() #check the / mark in the word
			key=key[key.index('.')+1:]
		else:
			if entity.find(' ')>0:
				entity=entity[:entity.index(' ')]
			key='m.'+entity
	#	pdb.set_trace()
		sql_subject='SELECT ?s ?p  WHERE {?s ?p <http://rdf.freebase.com/ns/'+key+'>} limit 3'
		sql_object='SELECT ?p ?o WHERE {<http://rdf.freebase.com/ns/'+key+'> ?p ?o} limit 3'
		result_sub[entity]=list(fb.query(PREFIX+sql_subject))
		result_obj[entity]=list(fb.query(PREFIX+sql_object))
			#get a random truple as result for test, to delete following  line in practice 
			#result_sub[key]=list(fb.query('select ?s ?p where {?s ?p ?o} limit 3'))
	fb.close()
	if len(result_sub)>0 or len(result_obj)>0:
		result['subject']=result_sub  #search subject by object
		result['object']=result_obj
#	print(result)
	return result

def get_additional_info(code_eng,result):
	if (len(result)==0):
		return [[]],[[]]
	addition_stories=[] # all result of queries
	triple_addition_stories=[]
	subject_result=result['subject'] # here we  do the subject of result
	addition_memory_triples=[]
	for key_object,value_subj_pred in subject_result.items():
		addition_memory=[]  #several results of one query
		if len(value_subj_pred)==0:
			continue
		while(len(value_subj_pred)>0):
			info_list=value_subj_pred.pop(0)
	#		pdb.set_trace()
			subject=split_out_key_word(str(info_list[0]))
			relation=split_out_key_word(str(info_list[1]))
#			relation=str(info_list[1])
	#		pdb.set_trace() #check the mark / in the word with map keys
			if code_eng.has_key(subject):
				subject=code_eng.get(subject)
			addition_memory.append(subject+' '+relation+' '+key_object)
			addition_memory_triples.append([subject,relation, key_object])	
		addition_stories.append(addition_memory)
	#print addition_stories
	object_result=result['object'] # here we do the object of result
	for key_subject,value_pred_obj in object_result.items():
		addition_memory=[]  #several results of one query
		if len(value_pred_obj)==0:
			continue
		while(len(value_pred_obj)>0):
			info_list=value_pred_obj.pop(0)
	#		pdb.set_trace()
	#		print (info_list[1])
			relation=str(info_list[0])
			object_=str(info_list[1])
			relation=split_out_key_word(relation)
			object_=split_out_key_word(object_)
	#		pdb.set_trace() #check the mark / in the word with map keys
			if code_eng.has_key(object_):
				object_=code_eng.get(object_)
			addition_memory.append(key_subject+' '+relation+' '+object_)
			addition_memory_triples.append([key_subject,relation, object_])	
		addition_stories.append(addition_memory)
	return addition_stories,addition_memory_triples
def split_out_key_word(object_):		
	if object_.rfind('m.')>=0:
		object_=object_[object_.rindex('m.')+2:]
#	if object_.rfind('#')>=0:
#		object_=object_[object_.rindex('#')+1:]
	if object_.rfind('/')>=0:
		object_=object_[object_.rindex('/')+1:]
	return object_

if __name__ =='__main__':
	fb_map=get_code_to_english('/data/analysis/qichuan_data/freebase/mid2name.txt')
	result=read_rdf_kb(fb_map[1],['Sandra',])
	get_additional_info(fb_map[0],result)

