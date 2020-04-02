# Для использования только на ЛД
import subprocess
import pandas as pd
import time


# Переводим байты в человеческий вид
def size_name(col):
    new_col = col.copy()
    new_col[col > 1024] = ['{0: .2f} KB'.format(round(x / 1024, 2)) for x in col[col > 1024]]
    new_col[col > 1024 * 1024] = ['{0: .2f} MB'.format(round(x / 1024 / 1024, 2)) for x in col[col > 1024 * 1024]]
    new_col[col > 1024 * 1024 * 1024] = ['{0: .2f} GB'.format(round(x / 1024 / 1024 / 1024, 2)) for x in col[col > 1024 * 1024 * 1024]]
    new_col[col > 1024 * 1024 * 1024 * 1024] = ['{0: .2f} TB'.format(round(x / 1024 / 1024 / 1024 / 1024, 2)) for x in col[col > 1024 * 1024 * 1024 * 1024]]
    return new_col


# Удаление таблиц или таблицы
def drop_tbl(tbls, verbose=1):
    if type(tbls) != list: query = 'hive -e "drop table {0};"'.format(';drop table '.join(tbls))
    elif type(tbls) == str: query = 'hive -e "drop table {0};"'.format(';drop table '.join([tbls]))
    else: raise Exception('Unknown argument type')
    start_time = time.time()
    script = subprocess.Popen('{0}'.format(query), shell=True, universal_newlines=True, stderr=subprocess.PIPE, )
    while script.poll() is None:
        time.sleep(1)
        if verbose == 1: print('Not ready ..{}'.format(round(time.time() - start_time)), end='\r')
        if (time.time() - start_time) > 60 * 1:
            if verbose == 1: print('Not ready in {0} sec. Timeout error'.format(round(time.time() - start_time)))
            return 1
    answer = script.communicate()[0]
    if verbose == 1: print('Ready in {0} sec \n{1}'.format(round(time.time() - start_time), answer))
    if script.poll() != 0:
        return answer
    return 0


# Удаление директории или директорий на ЛД
def del_folder(paths, verbose=1):
    if type(paths) == list: query = 'hdfs dfs -rm -r {0};'.format(';hdfs dfs -rm -r '.join(paths))
    elif type(paths) == str: query = 'hdfs dfs -rm -r {0};'.format(';hdfs dfs -rm -r '.join([paths]))
    else: raise Exception('Unknown argument type')
    start_time = time.time()
    script = subprocess.Popen('{0}'.format(query), shell=True, universal_newlines=True, stderr=subprocess.PIPE,)
    while script.poll() is None:
        time.sleep(1)
        if verbose == 1: print('Not ready ..{}'.format(round(time.time() - start_time)), end='\r')
        if (time.time() - start_time) > 60*1:
            if verbose == 1: print('Not ready in {0} sec. Timeout error'.format(round(time.time() - start_time)))
            return 1
    answer = script.communicate()[0]
    if verbose == 1: print('Ready in {0} sec \n{1}'.format(round(time.time() - start_time), answer))
    if script.poll() != 0:
        return answer
    return 0


# Изменение уровня репликации папки и всех файлов внутри
def set_repl_level(replication_level, path, verbose=1):
    query = 'hadoop fs -setrep -w {0} {1};'.format(replication_level, path)
    start_time = time.time()
    script = subprocess.Popen('{0}'.format(query), shell=True, universal_newlines=True, stderr=subprocess.PIPE, )
    while script.poll() is None:
        time.sleep(1)
        if verbose == 1: print('Not ready ..{}'.format(round(time.time() - start_time)), end='\r')
        if (time.time() - start_time) > 60 * 10:
            if verbose == 1: print('Not ready in {0} sec. Timeout error'.format(round(time.time() - start_time), ))
            return 1
    answer = script.communicate()[0]
    if verbose == 1: print('Ready in {0} sec \n{1}'.format(round(time.time() - start_time), answer))
    if script.poll() != 0:
        return answer
    return 0


def get_schemas_path(schemas):
    if type(schemas) == list: query = 'hive -e "describe database {0}";'.format(';describe database '.join(schemas))
    elif type(schemas) == str: query = 'hive -e "describe database {0}";'.format(';describe database '.join([schemas]))
    script = subprocess.Popen(
        '{0}'.format(query),
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,)
    answer = script.communicate()
    paths = [x.split('\t')[2] for x in answer[0].split('\n') if x != '']
    if script.poll() == 0: return paths[0] if len(paths) == 1 else paths
    else: return answer[1]


def describe_ext_table(schema, tbls, init_user=[], init_date=[], verbose=1):
    if type(tbls) != list: tbls = [tbls]
    if len(tbls) == 0: return init_user, init_date
    query = 'hive -e "describe extended {0};"'.format(';describe extended '\
                                                      .join(['{0}.{1}'.format(schema, tbl) for tbl in tbls])).replace('#', '_')
    p = subprocess.Popen(
        '{0} | grep -Po "(owner:|transient_lastDdlTime=).*?(?=,)"'.format(query),
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,)
    data = p.communicate()
    answer, errors = data[0], data[1]
    user = init_user + [x.split(':')[-1] for x in answer.split('\n') if 'owner:' in x]
    date = init_date + [x.split('=')[-1] for x in answer.split('\n') if 'transient_lastDdlTime=' in x]
    error_tbl = [x for x in errors.split('\n') if 'FAILED: ' in x]
    if len(error_tbl) > 0:
        if verbose == 1: print(error_tbl[0])
        user.append(None)
        date.append(None)
        return describe_ext_table(schema, tbls[len(user) - len(init_user):], init_user=user, init_date=date)
    return user, date


def get_all_folders_in_dir(directory):
    query = 'hadoop fs -du {0}'.format(directory)
    script = subprocess.Popen(
        '{0}'.format(query),
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,)
    answer = script.communicate()
    if script.poll() == 0: return answer[0]
    else: return answer[1]


# Парсим список таблиц из схемы и их параметры
# выводит либо все либо top таблиц, занимающих наибольшее дисковое пространство
# Режим fast не парсит поля user и date
def get_tables(schema, top=None, fast=False, sort=None, verbose=1):
    schema_path = get_schemas_path(schema)
    tables = pd.DataFrame(
        [x.split() for x in get_all_folders_in_dir(schema_path).split('\n') if x != ''],
        columns=['realSize', 'sizeOnDisk', 'path'])
    tables['realSize'] = tables['realSize'].astype('int')
    tables['sizeOnDisk'] = tables['sizeOnDisk'].astype('int')
    tables['replLevel'] = tables['sizeOnDisk'] / tables['realSize']
    tables['realSizeHuman'] = size_name(tables['realSize'])
    tables['sizeOnDiskHuman'] = size_name(tables['sizeOnDisk'])
    tables['path'] = tables['path'].str.split('/').apply(lambda x: x[-1])

    if top is None: top = tables[tables['sizeOnDisk'] != 0].shape[0]
    if sort is not None: tables = tables[tables['sizeOnDisk'] != 0].sort_values(sort, ascending=False).head(top)
    else: tables = tables[tables['sizeOnDisk'] != 0].sort_values('sizeOnDisk', ascending=False).head(top)

    if fast is False:
        user = list()
        date = list()
        tbls = tables.path.values.tolist()
        step = 2000
        for i in range(0, len(tbls) + step, step):
            start_point = i
            end_point = start_point + step
            if end_point > len(tbls):
                end_point = len(tbls)
            user_sub, date_sub = describe_ext_table(schema, tbls[start_point:end_point])
            user += user_sub
            date += date_sub
        tables['user'] = user
        tables['date'] = date
        tables['date'] = pd.to_datetime(tables.date.astype(int), unit='s')
    if verbose == 1: print('Директория {}'.format(schema_path))
    if verbose == 1: print('Данные таблицы занимают {0: .2f} TB'.format(tables['sizeOnDisk'].sum() / 1024 / 1024 / 1024 / 1024))
    return tables


def get_schemas():
    p = subprocess.Popen(
        'hive -e "show databases;"',
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,)
    data = p.communicate()
    if p.poll() != 0: return data[1]
    dbs = [x for x in data[0].split('\n') if x != '']
    db_paths = list(zip(dbs, get_schemas_path(dbs)))
    dbs_rawinfo = list(zip(db_paths, [get_all_folders_in_dir(db_path[1], s=True) for db_path in db_paths]))
    dbs_info = [[x[0][0], x[0][1], x[1].split()[0], x[1].split()[1]] for x in dbs_rawinfo if x[1] != 1]
    schemas = pd.DataFrame(dbs_info, columns=['schema', 'path', 'realSize', 'sizeOnDisk'])
    schemas['sizeOnDisk'] = schemas['sizeOnDisk'].astype(int)
    schemas['realSize'] = schemas['realSize'].astype(int)
    schemas['replLevel'] = schemas['sizeOnDisk'] / schemas['realSize']
    schemas['realSizeHuman'] = size_name(schemas['realSize'])
    schemas['sizeOnDiskHuman'] = size_name(schemas['sizeOnDisk'])
    return schemas
