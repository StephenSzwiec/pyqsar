#-*- coding: utf-8 -*-
# control SDF

def count_mol(filename) :
    """
    Count number of molecule in sdf file

    Parameters
    ----------
    filename : sdf file

    Returns
    -------
    None
    """
    f = open(filename)
    cnt = 0
    for i in f :
        if '$$$$' in i :
            cnt = cnt+1
    print('Number of mol in SDF :',cnt)

def cut_sdf(inputfile,outfile,n):
    """
    Save up to 'n' molecules separately as a new file

    Parameters
    ----------
    inputfile : original sdf file
    outfile : splited sdf file
    n : number of molecule what user want to split

    Returns
    -------
    None
    """
    f = open(inputfile)
    fw = open(outfile,'w')
    count = 0
    sdf = ''
    for line in f :
        sdf = sdf + line
        if '$$$$' in line :
            count += 1
            fw.write(sdf)
            sdf = ''
        if count == n :
            break

def remove_tail(inputfile,outfile):
    """
    Remove unnecessary informatioin of each molecule

    Parameters
    ----------
    filename : sdf file
    outfile : sdf file that removed unnecessary informatioin

    Returns
    -------
    None
    """
    inf = open(inputfile)
    outf = open(outfile,'w')
    sdf = ''
    for line in inf :
        sdf = sdf + line
        if '$$$$' in line :
            sdf = sdf.split('M  END')[0]
            sdf = sdf + 'M END\n' +'$$$$\n'
            outf.write(sdf)
            sdf = ''
